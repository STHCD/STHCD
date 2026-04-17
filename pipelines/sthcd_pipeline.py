from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from utils.runtime_warnings import suppress_optional_dependency_warnings

suppress_optional_dependency_warnings()

import torch
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import retrieve_latents, retrieve_timesteps
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.utils import deprecate
from typing import Any, Callable, Dict, List, Optional, Union
import PIL

from utils.index_history import load_batch_histories, save_scheduler_index_history, stack_step_history


class STHCDPipeline(StableDiffusionImg2ImgPipeline):
    """Stable Diffusion image-to-image pipeline adapted for STHCD."""

    @staticmethod
    def _normalize_progressive_stages(
        progressive_reconstruction_stages,
        num_inference_steps: int,
    ) -> set[int]:
        if progressive_reconstruction_stages in (None, "", []):
            return set()

        if isinstance(progressive_reconstruction_stages, str):
            raw_items = progressive_reconstruction_stages.split(",")
        else:
            raw_items = progressive_reconstruction_stages

        normalized: set[int] = set()
        for raw_item in raw_items:
            item = str(raw_item).strip()
            if not item:
                continue
            stage = int(item)
            if stage < 1 or stage > int(num_inference_steps):
                raise ValueError(
                    f"progressive reconstruction stage {stage} is outside [1, {num_inference_steps}]."
                )
            normalized.add(stage)
        return normalized

    @staticmethod
    def _should_save_progressive_reconstruction(
        step_number: int,
        num_inference_steps: int,
        stage_interval: int | None,
        explicit_stages: set[int],
    ) -> bool:
        if step_number == num_inference_steps:
            return True

        if explicit_stages and step_number in explicit_stages:
            return True

        if stage_interval is not None and stage_interval > 0 and step_number % stage_interval == 0:
            return True

        if not explicit_stages and stage_interval is None:
            auto_interval = max(1, int(num_inference_steps) // 10)
            return step_number == 1 or step_number % auto_interval == 0

        return False

    def prepare_sthcd_latents(self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        image = image.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt

        if image.shape[1] == 4:
            init_latents = image

        else:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            elif isinstance(generator, list):
                if image.shape[0] < batch_size and batch_size % image.shape[0] == 0:
                    image = torch.cat([image] * (batch_size // image.shape[0]), dim=0)
                elif image.shape[0] < batch_size and batch_size % image.shape[0] != 0:
                    raise ValueError(
                        f"Cannot duplicate `image` of batch size {image.shape[0]} to effective batch_size {batch_size} "
                    )

                init_latents = [
                    retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                    for i in range(batch_size)
                ]
                init_latents = torch.cat(init_latents, dim=0)
            else:
                init_latents = retrieve_latents(self.vae.encode(image), generator=generator)

            init_latents = self.vae.config.scaling_factor * init_latents

        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            # expand init_latents for batch_size
            deprecation_message = (
                f"You have passed {batch_size} text prompts (`prompt`), but only {init_latents.shape[0]} initial"
                " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                " your script to pass as many initial images as text prompts to suppress this warning."
            )
            deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents], dim=0)

        latents = init_latents

        return latents

    # Adapted from diffusers.pipelines.stable_diffusion.StableDiffusionImg2ImgPipeline.__call__
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: PipelineImageInput = None,
        strength: float = 1.0,
        num_inference_steps: Optional[int] = 1000,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: Optional[float] = 0.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 1.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: int = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        # Runtime mode: "codec", "encode", or "decode"
        mode: str = "codec",
        # Shared parameters
        grad_save=False,
        grad_dir=None,
        save_progressive_reconstruction: bool = False,
        progressive_reconstruction_dir: Optional[str] = None,
        progressive_reconstruction_interval: Optional[int] = None,
        progressive_reconstruction_stages: Optional[Union[str, List[int]]] = None,
        batch_id=None,
        return_step_metrics=False,
        # Encode/decode parameters
        bin_dir=None,
        **kwargs,
    ):
        """
        Unified call method that supports multiple modes:
        - codec: Standard diffusion pipeline (original __call__ behavior)
        - encode: Encode images and save index history to bin files
        - decode: Decode images from saved index history
        """

        # Validate mode
        valid_modes = ["codec", "encode", "decode"]
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got {mode}")

        if grad_save:
            save_progressive_reconstruction = True
        if grad_dir is not None and progressive_reconstruction_dir is None:
            progressive_reconstruction_dir = grad_dir
        if progressive_reconstruction_interval is not None and progressive_reconstruction_interval <= 0:
            raise ValueError("progressive_reconstruction_interval must be positive when provided.")

        # Mode-specific validation
        if mode == "decode" and (bin_dir is None or batch_id is None):
            raise ValueError(f"bin_dir and batch_id must be provided for mode '{mode}'")

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        self.check_inputs(
            prompt,
            strength,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        loaded_index_histories = None
        explicit_progressive_stages = self._normalize_progressive_stages(
            progressive_reconstruction_stages,
            num_inference_steps,
        )
        self.progressive_reconstruction_records = []

        text_encoder_lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self.clip_skip,
        )

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        if mode in ["codec", "encode"]:
            image = self.image_processor.preprocess(image)

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)

        if mode == "decode":
            loaded_index_histories = load_batch_histories(
                bin_dir=bin_dir,
                batch_ids=batch_id,
                scheduler=self.scheduler,
                timesteps=timesteps,
                codebook_size=self.scheduler.codebook_size,
            )
            if not all(len(hist) == len(loaded_index_histories[0]) for hist in loaded_index_histories):
                raise ValueError("All index histories must have the same length")

        if mode in ["codec", "encode"]:
            latent_timestep = (timesteps[:1]-1).repeat(batch_size * num_images_per_prompt)
            x0_latents = self.prepare_sthcd_latents(
                image,
                latent_timestep,
                batch_size,
                num_images_per_prompt,
                prompt_embeds.dtype,
                device,
                generator,
            )

        rand_indices = torch.full((batch_size,), self.scheduler.codebook_size-1, dtype=int)
        latents = self.scheduler.codebook[self.scheduler.codebook_entries-1][rand_indices].to(device=device)

        # Handle multi-channel codebook case
        if latents.shape[1] > 4:
            latents = latents[:, :4, :, :]
        elif latents.shape[1] < 4:
            # Handle specific cases for 1, 2, and 3 channels
            if latents.shape[1] == 1:
                # Repeat the single channel 4 times
                latents = latents.repeat(1, 4, 1, 1)
            elif latents.shape[1] == 2:
                # Repeat each channel 2 times
                latents = latents.repeat(1, 2, 1, 1)
            elif latents.shape[1] == 3:
                # Copy the first channel once to make 4 channels
                additional_channel = latents[:, :1, :, :]
                latents = torch.cat([latents, additional_channel], dim=1)
        
        # Handle size adjustment to target shape (batch, 4, 32, 32)
        target_h, target_w = 32, 32  # Default target dimensions
        if mode in ["codec", "encode"] and x0_latents is not None:
            # Use x0_latents dimensions as target
            target_h, target_w = x0_latents.shape[-2:]

        current_h, current_w = latents.shape[-2:]

        # Check if we need to resize latents
        if current_h != target_h or current_w != target_w:
            # Validate that target dimensions are compatible
            if target_h % current_h != 0 or target_w % current_w != 0:
                raise ValueError(
                    f"Target dimensions ({target_h}, {target_w}) must be integer multiples "
                    f"of current latent dimensions ({current_h}, {current_w}). "
                    f"Got ratios: h={target_h/current_h}, w={target_w/current_w}"
                )
            
            # Calculate replication factors
            h_factor = target_h // current_h
            w_factor = target_w // current_w
            
            # Replicate latents to match target size
            if h_factor > 1:
                latents = latents.repeat_interleave(h_factor, dim=-2)
            if w_factor > 1:
                latents = latents.repeat_interleave(w_factor, dim=-1)

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if ip_adapter_image is not None or ip_adapter_image_embeds is not None
            else None
        )

        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self.scheduler.reset_index_history()
        self._num_timesteps = len(timesteps)
        
        # Initialize step metrics collection
        step_metrics = [] if return_step_metrics else None


        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                if mode in ["codec", "encode"]:
                    # Standard step with x0_latents
                    latents, latents_pred_x0 = self.scheduler.step(
                        noise_pred, t, latents, **extra_step_kwargs, return_dict=False, x0_latents=x0_latents
                    )
                else:  # decode mode
                    idx_sim = [hist[i] for hist in loaded_index_histories]
                    
                    latents, latents_pred_x0 = self.scheduler.decode_step(
                        model_output=noise_pred,
                        timestep=t,
                        sample=latents,
                        idx_codebook=stack_step_history(idx_sim, device=device),
                        **extra_step_kwargs,
                        return_dict=False
                    )

                # Collect step metrics if requested
                if return_step_metrics:
                    t_step = int((t.item()) * num_inference_steps / 1000)
                    should_record = (i % 1 == 0) or (i == len(timesteps) - 1) 
                    
                    if should_record:
                        step_metric = {
                            'step': i,
                            'timestep': t.item(),
                            'timestep_normalized': t_step
                        }
                        
                        # Calculate difference for codec mode
                        if mode in ["codec", "encode"]:
                            diff = torch.abs(latents_pred_x0 - x0_latents.to(device=device, dtype=latents_pred_x0.dtype))
                            step_metric['mean_diff'] = diff.mean().item()
                        
                        # Store intermediate images
                        decode_latents = latents if t_step == 0 else latents_pred_x0
                        with torch.no_grad():
                            interim_img = self.vae.decode(decode_latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
                        interim_img_pil = self.image_processor.postprocess(interim_img, output_type="pil", do_denormalize=[True] * interim_img.shape[0])
                        
                        step_metric['has_image_metrics'] = True
                        step_metric['interim_images'] = interim_img_pil if isinstance(interim_img_pil, list) else [interim_img_pil]
                        step_metrics.append(step_metric)

                # Save intermediate images if requested
                if save_progressive_reconstruction and progressive_reconstruction_dir is not None:
                    step_number = i + 1
                    if self._should_save_progressive_reconstruction(
                        step_number=step_number,
                        num_inference_steps=num_inference_steps,
                        stage_interval=progressive_reconstruction_interval,
                        explicit_stages=explicit_progressive_stages,
                    ):
                        timestep_value = int(t.item())
                        decode_latents = latents if step_number == num_inference_steps else latents_pred_x0
                        with torch.no_grad():
                            progressive_images = self.vae.decode(
                                decode_latents / self.vae.config.scaling_factor,
                                return_dict=False,
                                generator=generator,
                            )[0]
                        progressive_images = self.image_processor.postprocess(
                            progressive_images,
                            output_type="pil",
                            do_denormalize=[True] * progressive_images.shape[0],
                        )

                        resolved_batch_ids = (
                            list(batch_id)
                            if batch_id is not None and len(batch_id) == len(progressive_images)
                            else [f"sample_{batch_idx}" for batch_idx in range(len(progressive_images))]
                        )
                        base_dir = Path(progressive_reconstruction_dir)
                        for image_obj, image_id in zip(progressive_images, resolved_batch_ids):
                            image_dir = base_dir / str(image_id)
                            image_dir.mkdir(parents=True, exist_ok=True)
                            file_name = f"stage_{step_number:03d}_t{timestep_value:04d}.png"
                            file_path = image_dir / file_name
                            image_obj.save(file_path)
                            self.progressive_reconstruction_records.append(
                                {
                                    "mode": mode,
                                    "image_id": str(image_id),
                                    "stage": int(step_number),
                                    "timestep": timestep_value,
                                    "relative_progress": float(step_number) / float(num_inference_steps),
                                    "file_name": file_name,
                                    "output_path": str(Path(str(image_id)) / file_name),
                                }
                            )

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        # Save binary files for encode mode
        if mode == "encode" and bin_dir is not None:
            self._save_encode_indices(batch_size, len(timesteps), bin_dir, batch_id)

        # Final image processing
        if not output_type == "latent":
            image = self.vae.decode(latents_pred_x0 / self.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents_pred_x0
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        # Return results
        if not return_dict:
            result = (image, has_nsfw_concept)
            if return_step_metrics:
                class SimpleOutput:
                    def __init__(self, images, nsfw, step_metrics):
                        self.images = images
                        self.nsfw_content_detected = nsfw
                        self.step_metrics = step_metrics
                return SimpleOutput(image, has_nsfw_concept, step_metrics)
            return result

        output = StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
        if return_step_metrics:
            output.step_metrics = step_metrics
        return output


    def _save_encode_indices(self, batch_size, num_timesteps, bin_dir, batch_id):
        if not hasattr(self.scheduler, 'index_history') or not self.scheduler.index_history:
            print("Warning: No quantization history found in scheduler.")
            return
        resolved_batch_ids = (
            list(batch_id)
            if batch_id is not None
            else [f"sample_{batch_idx}" for batch_idx in range(batch_size)]
        )
        save_scheduler_index_history(
            bin_dir=bin_dir,
            batch_ids=resolved_batch_ids,
            history=self.scheduler.index_history,
            num_timesteps=num_timesteps,
            batch_size=batch_size,
            codebook_size=self.scheduler.codebook_size,
            image_size=256,
            metadata={"sthcd": True},
        )
