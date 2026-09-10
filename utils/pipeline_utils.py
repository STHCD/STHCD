from __future__ import annotations

from utils.runtime_warnings import suppress_optional_dependency_warnings

suppress_optional_dependency_warnings()

import torch

from schedulers.residual_transform import ResidualTransformScheduler
from schedulers.channel_hierarchy import ChannelHierarchyScheduler
from schedulers.spatial_hierarchy import SpatialHierarchyScheduler
from schedulers.spatiotemporal_hierarchy import SpatiotemporalHierarchyScheduler

from pipelines.sthcd_pipeline import STHCDPipeline
from utils.schedule_registry import resolve_schedule_payload

SD21_MODEL_IDS = [
    "STHCD/stable-diffusion-2-1-base",
]


class HuggingFaceLoadError(RuntimeError):
    """Raised when required Hugging Face assets cannot be loaded."""


def _resolve_data_type(data_type: str | None = None) -> str:
    if data_type in (None, "float16"):
        return "float16"
    raise ValueError("STHCD only supports float16 runtime precision.")


def _as_model_id_candidates(model_id):
    if isinstance(model_id, str):
        return [model_id]
    return list(model_id)


def _load_pretrained_component(
    loader_cls,
    model_id,
    *,
    local_files_only=False,
    component_name=None,
    **kwargs,
):
    model_ids = _as_model_id_candidates(model_id)
    label = component_name or loader_cls.__name__
    subfolder = kwargs.get("subfolder")
    last_exc = None

    for current_model_id in model_ids:
        try:
            return loader_cls.from_pretrained(
                current_model_id,
                local_files_only=local_files_only,
                **kwargs,
            )
        except (OSError, EnvironmentError) as exc:
            last_exc = exc

    display_ids = ", ".join(model_ids)
    if subfolder:
        display_ids = ", ".join(f"{candidate}/{subfolder}" for candidate in model_ids)

    if local_files_only:
        raise HuggingFaceLoadError(
            f"Failed to load {label} from local cache. Tried: {display_ids}. "
            "The required Hugging Face files are not cached locally. "
            "Re-run without `--local-files-only`, or pre-download them with "
            "`python scripts/download_sd21.py`."
        ) from last_exc

    raise HuggingFaceLoadError(
        f"Failed to load {label} from Hugging Face. Tried: {display_ids}. "
        "Check your network / proxy settings, or pre-download the assets with "
        "`python scripts/download_sd21.py`."
    ) from last_exc


def _parse_ratio_schedule(schedule_data, field_name, num_timesteps):
    raw_map = schedule_data.get(field_name, {})
    return {int(float(k) * num_timesteps): int(v) for k, v in raw_map.items()}


def _parse_cosine_max_beta_map(schedule_data):
    raw_map = schedule_data.get("cosine_max_beta_map", {})
    return {eval(k): v for k, v in raw_map.items()}


def load_sd21_pipe(
    scheduler,
    device,
    data_type="float16",
    local_files_only=False,
):
    _resolve_data_type(data_type)
    dtype = torch.float16

    pipe = _load_pretrained_component(
        STHCDPipeline,
        SD21_MODEL_IDS,
        component_name="Stable Diffusion pipeline",
        scheduler=scheduler,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=local_files_only,
    ).to(device)
    return pipe


def load_residual_hierarchy_scheduler(
    num_timesteps=1000,
    codebook_size=64,
    codebook_dims=(4, 32, 32),
    data_type="float16",
    residual_unshuffle_factor=2,
    use_progressive_codebook_channel=False,
    channel_schedule_path=None,
    use_progressive_shuffle_factor=False,
    shuffle_schedule_path=None,
    channel_schedule_data=None,
    shuffle_schedule_data=None,
    local_files_only=False,
    **kwargs,
):
    """
    Load and configure the residual hierarchy scheduler family used by STHCD.
    """
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected keyword arguments: {unexpected}")
    data_type = _resolve_data_type(data_type)

    if use_progressive_shuffle_factor and use_progressive_codebook_channel:
        if not shuffle_schedule_path:
            raise ValueError("shuffle_factor_schedule_path must be provided when use_progressive_shuffle_factor is True.")
        combined_data = resolve_schedule_payload(
            shuffle_schedule_data if shuffle_schedule_data is not None else shuffle_schedule_path
        )
        shuffle_factor_schedule = _parse_ratio_schedule(
            combined_data, "shuffle_factor_map", num_timesteps
        )
        codebook_channel_schedule = _parse_ratio_schedule(
            combined_data, "codebook_channel_map", num_timesteps
        )
        cosine_max_beta_map = _parse_cosine_max_beta_map(combined_data)

        cosine_max_beta = cosine_max_beta_map.get((num_timesteps, residual_unshuffle_factor, codebook_dims[0]), None)

        scheduler = _load_pretrained_component(
            SpatiotemporalHierarchyScheduler,
            SD21_MODEL_IDS,
            component_name="Spatiotemporal hierarchy scheduler",
            subfolder="scheduler",
            cosine_max_beta=cosine_max_beta,
            codebook_size=codebook_size,
            codebook_entries=num_timesteps + 1,
            codebook_dims=codebook_dims,
            data_type=data_type,
            requires_safety_checker=False,
            local_files_only=local_files_only,
            beta_start=0.00085,
            beta_end=0.012,
            residual_unshuffle_factor=residual_unshuffle_factor,
            shuffle_factor_schedule=shuffle_factor_schedule,
            codebook_channel_schedule=codebook_channel_schedule,
        )
    elif use_progressive_codebook_channel:
        if not channel_schedule_path:
            raise ValueError("channel_schedule_path must be provided when use_progressive_codebook_channel is True.")
        combined_data = resolve_schedule_payload(
            channel_schedule_data if channel_schedule_data is not None else channel_schedule_path
        )
        codebook_channel_schedule = _parse_ratio_schedule(
            combined_data, "codebook_channel_map", num_timesteps
        )
        cosine_max_beta_map = _parse_cosine_max_beta_map(combined_data)

        cosine_max_beta = cosine_max_beta_map.get((num_timesteps, residual_unshuffle_factor, codebook_dims[0]), None)

        scheduler = _load_pretrained_component(
            ChannelHierarchyScheduler,
            SD21_MODEL_IDS,
            component_name="Channel hierarchy scheduler",
            subfolder="scheduler",
            cosine_max_beta=cosine_max_beta,
            codebook_size=codebook_size,
            codebook_entries=num_timesteps + 1,
            codebook_dims=codebook_dims,
            data_type=data_type,
            requires_safety_checker=False,
            local_files_only=local_files_only,
            beta_start=0.00085,
            beta_end=0.012,
            residual_unshuffle_factor=residual_unshuffle_factor,
            codebook_channel_schedule=codebook_channel_schedule,
        )
    elif use_progressive_shuffle_factor:
        if not shuffle_schedule_path:
            raise ValueError("shuffle_factor_schedule_path must be provided when use_progressive_shuffle_factor is True.")
        combined_data = resolve_schedule_payload(
            shuffle_schedule_data if shuffle_schedule_data is not None else shuffle_schedule_path
        )
        shuffle_factor_schedule = _parse_ratio_schedule(
            combined_data, "shuffle_factor_map", num_timesteps
        )
        cosine_max_beta_map = _parse_cosine_max_beta_map(combined_data)

        cosine_max_beta = cosine_max_beta_map.get((num_timesteps, residual_unshuffle_factor, codebook_dims[0]), None)

        scheduler = _load_pretrained_component(
            SpatialHierarchyScheduler,
            SD21_MODEL_IDS,
            component_name="Spatial hierarchy scheduler",
            subfolder="scheduler",
            cosine_max_beta=cosine_max_beta,
            codebook_size=codebook_size,
            codebook_entries=num_timesteps + 1,
            codebook_dims=codebook_dims,
            data_type=data_type,
            requires_safety_checker=False,
            local_files_only=local_files_only,
            beta_start=0.00085,
            beta_end=0.012,
            residual_unshuffle_factor=residual_unshuffle_factor,
            shuffle_factor_schedule=shuffle_factor_schedule,
        )
    else:
        print("INFO: Loading residual transform scheduler...")
        cosine_max_beta_map = _parse_cosine_max_beta_map(
            {"cosine_max_beta_map": resolve_schedule_payload("beta")}
        )

        cosine_max_beta = cosine_max_beta_map.get((num_timesteps, residual_unshuffle_factor, codebook_dims[0]), None)

        scheduler = _load_pretrained_component(
            ResidualTransformScheduler,
            SD21_MODEL_IDS,
            component_name="Residual transform scheduler",
            subfolder="scheduler",
            cosine_max_beta=cosine_max_beta,
            codebook_size=codebook_size,
            codebook_entries=num_timesteps + 1,
            codebook_dims=codebook_dims,
            data_type=data_type,
            requires_safety_checker=False,
            local_files_only=local_files_only,
            beta_start=0.00085,
            beta_end=0.012,
            residual_unshuffle_factor=residual_unshuffle_factor,
        )

    scheduler.config.timestep_spacing = "trailing"
    scheduler.set_timesteps(num_timesteps)
    return scheduler
