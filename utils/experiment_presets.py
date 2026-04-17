from __future__ import annotations

from copy import deepcopy
from typing import Any


PUBLIC_PROFILE_NAMES = ("baseline", "spatial", "channel", "sthcd")
PROFILE_ALIASES = {
    "shuffle": "spatial",
    "channel_shuffle": "sthcd",
}


GENERIC_DATA_DIR = "data/<remote_sensing_data_dir>"


MODALITY_DEFAULTS: dict[str, dict[str, Any]] = {
    "visible": {
        "image_dir": GENERIC_DATA_DIR,
        "original_image_dir": GENERIC_DATA_DIR,
        "batch_size": 1,
        "num_workers": 0,
    },
    "sar": {
        "image_dir": GENERIC_DATA_DIR,
        "original_image_dir": GENERIC_DATA_DIR,
        "batch_size": 64,
        "num_workers": 0,
    },
    "hsi": {
        "image_dir": GENERIC_DATA_DIR,
        "original_image_dir": GENERIC_DATA_DIR,
        "batch_size": 64,
        "num_workers": 0,
    },
}


RATE_PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "baseline": {
        "low": {
            "num_timesteps": 200,
            "residual_unshuffle_factor": 1,
            "codebook_channel": 4,
            "schedule_ref": None,
        },
        "mid": {
            "num_timesteps": 500,
            "residual_unshuffle_factor": 1,
            "codebook_channel": 4,
            "schedule_ref": None,
        },
        "high": {
            "num_timesteps": 1000,
            "residual_unshuffle_factor": 1,
            "codebook_channel": 4,
            "schedule_ref": None,
        },
    },
    "spatial": {
        "low": {
            "num_timesteps": 50,
            "residual_unshuffle_factor": 1,
            "codebook_channel": 4,
            "schedule_ref": "spatial",
        },
        "mid": {
            "num_timesteps": 100,
            "residual_unshuffle_factor": 1,
            "codebook_channel": 4,
            "schedule_ref": "spatial",
        },
        "high": {
            "num_timesteps": 200,
            "residual_unshuffle_factor": 1,
            "codebook_channel": 4,
            "schedule_ref": "spatial",
        },
    },
    "channel": {
        "low": {
            "num_timesteps": 100,
            "residual_unshuffle_factor": 2,
            "codebook_channel": 4,
            "schedule_ref": "channel",
        },
        "mid": {
            "num_timesteps": 200,
            "residual_unshuffle_factor": 2,
            "codebook_channel": 4,
            "schedule_ref": "channel",
        },
        "high": {
            "num_timesteps": 500,
            "residual_unshuffle_factor": 1,
            "codebook_channel": 4,
            "schedule_ref": "channel",
        },
    },
    "sthcd": {
        "low": {
            "num_timesteps": 64,
            "residual_unshuffle_factor": 1,
            "codebook_channel": 8,
            "schedule_ref": "sthcd",
        },
        "mid": {
            "num_timesteps": 80,
            "residual_unshuffle_factor": 1,
            "codebook_channel": 8,
            "schedule_ref": "sthcd",
        },
        "high": {
            "num_timesteps": 100,
            "residual_unshuffle_factor": 1,
            "codebook_channel": 8,
            "schedule_ref": "sthcd",
        },
    },
}


def get_modality_defaults(modality: str) -> dict[str, Any]:
    try:
        return deepcopy(MODALITY_DEFAULTS[modality])
    except KeyError as exc:
        raise ValueError(f"Unknown modality: {modality}") from exc


def normalize_profile_name(profile: str) -> str:
    return PROFILE_ALIASES.get(profile, profile)


def list_profile_choices() -> list[str]:
    return sorted(set(PUBLIC_PROFILE_NAMES) | set(PROFILE_ALIASES))


def resolve_rate_preset(profile: str, rate: str) -> dict[str, Any]:
    profile = normalize_profile_name(profile)
    try:
        preset = RATE_PRESETS[profile][rate]
    except KeyError as exc:
        raise ValueError(f"Unknown preset profile/rate: {profile}/{rate}") from exc

    resolved = deepcopy(preset)
    resolved["profile"] = profile
    resolved["rate"] = rate
    return resolved


def list_rate_presets() -> dict[str, list[str]]:
    return {profile: sorted(presets) for profile, presets in RATE_PRESETS.items()}
