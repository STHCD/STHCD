import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, Optional, Tuple, Union

import torch
from torch.nn import functional as F

from schedulers.original_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMSchedulerOutput

from utils.codebook import DynamicGaussianCodebookWithCache
from utils.residual_decode import reconstruct_quantized_noise
from utils.residual_transform import residual_restore, residual_unshuffle
from utils.runtime_paths import resolve_codebook_cache_dir


class SpatialHierarchyScheduler(DDPMScheduler):
    def __init__(
        self,
        cosine_max_beta: Optional[float] = None,
        codebook_size: int = 256,
        codebook_entries: int = 1001,
        codebook_dims: Tuple[int] = (4, 32, 32),
        data_type: str = "float16",
        residual_unshuffle_factor: Optional[int] = None,
        shuffle_factor_schedule: Optional[Dict[int, int]] = None,
        *args,
        **kwargs,
    ):
        if residual_unshuffle_factor is None:
            residual_unshuffle_factor = 2
        if data_type not in (None, "float16"):
            raise ValueError("STHCD only supports float16 scheduler precision.")

        super().__init__(cosine_max_beta=cosine_max_beta, *args, **kwargs)
        self.codebook_size = codebook_size
        self.codebook_entries = codebook_entries
        self.codebook_dims = codebook_dims
        self.data_type = "float16"
        self.dtype = torch.float16

        self._init_codebook(load_codebook=True)
        self.index_history = []

        self.residual_unshuffle_factor = residual_unshuffle_factor

        if shuffle_factor_schedule:
            self.shuffle_factor_schedule = shuffle_factor_schedule
        else:
            self.shuffle_factor_schedule = {0: self.residual_unshuffle_factor}

    def _init_codebook(self, load_codebook=False):
        if not load_codebook:
            self.codebook = torch.randn(
                self.codebook_entries,
                self.codebook_size,
                *self.codebook_dims,
                dtype=self.dtype,
            )
        else:
            cache_name = (
                f"residual_transform_codebook_cache_{self.codebook_size}_{self.codebook_entries-1}_"
                f"{self.codebook_dims[-3]}_{self.codebook_dims[-2]}_{self.codebook_dims[-1]}"
            )
            self.codebook = DynamicGaussianCodebookWithCache(
                num_timesteps=self.codebook_entries - 1,
                codebook_size=self.codebook_size,
                image_shape=self.codebook_dims,
                batch_size=self.codebook_entries,
                cache_dir=resolve_codebook_cache_dir(
                    self.data_type,
                    "residual_transform_codebook",
                    cache_name,
                ),
                data_type=self.data_type,
                load_on_init=True,
            )

    def reset_index_history(self):
        self.index_history = []

    def _get_current_shuffle_config(self, step_index: int) -> int:
        for threshold, k in self.shuffle_factor_schedule.items():
            if step_index >= threshold:
                return k
        return list(self.shuffle_factor_schedule.values())[-1]

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
        x0_latents: Optional[torch.Tensor] = None,
    ) -> Union[DDPMSchedulerOutput, Tuple]:
        t = timestep
        prev_t = self.previous_timestep(t)

        if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in ["learned", "learned_range"]:
            model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
        else:
            predicted_variance = None

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
                " `v_prediction`  for the DDPMScheduler."
            )

        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

        residual = x0_latents - pred_original_sample
        t_index = timestep.item() if isinstance(timestep, torch.Tensor) else timestep
        step_index = int(t_index * self.num_inference_steps / 1000)
        codebook_t = self.codebook[step_index].to(device=model_output.device)

        current_shuffle_factor = self._get_current_shuffle_config(step_index)
        transformed_residual = residual_unshuffle(
            residual,
            residual.shape[-2] // current_shuffle_factor,
            residual.shape[-1] // current_shuffle_factor,
        )

        _, transformed_channels, residual_height, residual_width = transformed_residual.shape
        _, codebook_channels, codebook_height, codebook_width = codebook_t.shape
        if (residual_height != codebook_height) or (residual_width != codebook_width):
            if residual_height < codebook_height and residual_width < codebook_width:
                codebook_t = codebook_t[:, :, :residual_height, :residual_width]
            elif residual_height > codebook_height and residual_width > codebook_width:
                codebook_t = F.interpolate(
                    codebook_t, size=(residual_height, residual_width), mode="bilinear", align_corners=False
                )
            else:
                raise ValueError(
                    "Spatial mismatch after residual unshuffle: "
                    f"(H,W)={(residual_height, residual_width)} vs codebook {(codebook_height, codebook_width)}"
                )
        _, codebook_channels, _, _ = codebook_t.shape
        channel_groups = transformed_channels // codebook_channels

        def _quantize_slice(x_slice: torch.Tensor):
            sims = torch.einsum("kcwh,bcwh->bk", codebook_t, x_slice)
            idx = sims.argmax(dim=1)
            quant = codebook_t[idx]
            return quant, idx

        if channel_groups == 1:
            z_tilde, idx = _quantize_slice(transformed_residual)
            idx_all = idx.unsqueeze(1)
        else:
            z_slices = []
            idx_slices = []
            for i in range(channel_groups):
                x_slice = transformed_residual[
                    :, i * codebook_channels : (i + 1) * codebook_channels, :, :
                ]
                quant_i, idx_i = _quantize_slice(x_slice)
                z_slices.append(quant_i)
                idx_slices.append(idx_i)
            z_tilde = torch.cat(z_slices, dim=1)
            idx_all = torch.stack(idx_slices, dim=1)

        if not hasattr(self, "index_history"):
            self.index_history = []
        self.index_history.append(idx_all)

        quantized_noise = residual_restore(
            z_tilde,
            z_tilde.shape[-2] * current_shuffle_factor,
            z_tilde.shape[-1] * current_shuffle_factor,
        )

        variance = 0
        if t > 0:
            device = model_output.device
            variance_noise = quantized_noise.to(device=device)

            if self.variance_type == "fixed_small_log":
                variance = self._get_variance(t, predicted_variance=predicted_variance) * variance_noise
            elif self.variance_type == "learned_range":
                variance = self._get_variance(t, predicted_variance=predicted_variance)
                variance = torch.exp(0.5 * variance) * variance_noise
            else:
                variance = (self._get_variance(t, predicted_variance=predicted_variance) ** 0.5) * variance_noise

        pred_prev_sample = pred_prev_sample + variance

        if not return_dict:
            return (pred_prev_sample, pred_original_sample)

        return DDPMSchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample)

    def decode_step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        idx_codebook,
        generator=None,
        return_dict: bool = True,
    ) -> Union[DDPMSchedulerOutput, Tuple]:
        t = timestep
        prev_t = self.previous_timestep(t)

        if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in ["learned", "learned_range"]:
            model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
        else:
            predicted_variance = None

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
                " `v_prediction` for the DDPMScheduler."
            )

        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

        t_index = timestep.item() if isinstance(timestep, torch.Tensor) else timestep
        step_index = int((t_index) * self.num_inference_steps / 1000)
        codebook_t = self.codebook[step_index].to(device=model_output.device)
        current_shuffle_factor = self._get_current_shuffle_config(step_index)
        quantized_noise = reconstruct_quantized_noise(
            idx_codebook=torch.as_tensor(idx_codebook, device=model_output.device, dtype=torch.long),
            codebook_t=codebook_t,
            sample_shape=sample.shape,
            residual_factor=current_shuffle_factor,
            allow_spatial_resize=True,
            slide_stride=getattr(self, "slide_stride", 1),
            prefer_ch0_when_sliding=getattr(self, "prefer_ch0_when_sliding", True),
        )

        variance = 0
        if t > 0:
            variance_noise = quantized_noise.to(device=model_output.device)

            if self.variance_type == "fixed_small_log":
                variance = self._get_variance(t, predicted_variance=predicted_variance) * variance_noise
            elif self.variance_type == "learned_range":
                variance = self._get_variance(t, predicted_variance=predicted_variance)
                variance = torch.exp(0.5 * variance) * variance_noise
            else:
                variance = (self._get_variance(t, predicted_variance=predicted_variance) ** 0.5) * variance_noise

        pred_prev_sample = pred_prev_sample + variance

        if not return_dict:
            return (pred_prev_sample, pred_original_sample)

        return DDPMSchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample)
