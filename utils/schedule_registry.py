from __future__ import annotations

import json
import zlib
from pathlib import Path
from typing import Any, Dict

from utils.runtime_warnings import suppress_optional_dependency_warnings

suppress_optional_dependency_warnings()

import torch

from utils.runtime_paths import get_schedule_bundle_path


PUBLIC_SCHEDULE_ALIASES: dict[str, str] = {
    "beta": "cosine_max_beta_map",
    "spatial": "progressive/progressive_shuffle_factor_map_2stages_2_4",
    "channel": "progressive/progressive_codebook_channel_map_3stages_4_2_1",
    "sthcd": "progressive_channel_shuffle/progressive_shuffle_2stages_2_4_channel_3stages_8_4_2",
}


SCHEDULE_REF_ALIASES: dict[str, str] = {
    **PUBLIC_SCHEDULE_ALIASES,
    "spatial_hierarchy": PUBLIC_SCHEDULE_ALIASES["spatial"],
    "channel_hierarchy": PUBLIC_SCHEDULE_ALIASES["channel"],
    "spatiotemporal_hierarchy": PUBLIC_SCHEDULE_ALIASES["sthcd"],
    "channel_shuffle": PUBLIC_SCHEDULE_ALIASES["sthcd"],
    "shuffle": PUBLIC_SCHEDULE_ALIASES["spatial"],
}


def _build_aliases(relpath: str) -> list[str]:
    path = Path(relpath)
    stem_path = path.with_suffix("").as_posix()
    aliases = {
        stem_path,
        path.name,
        path.stem,
    }
    return sorted(alias for alias in aliases if alias)


def _normalize_schedule_ref(schedule_ref: str) -> str:
    normalized = schedule_ref.replace("\\", "/").strip()
    for prefix in (
        "data/schedules/",
        "resources/schedule_sources/",
        "schedule_sources/",
    ):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break

    if normalized.endswith(".json"):
        normalized = normalized[:-5]
    return SCHEDULE_REF_ALIASES.get(normalized, normalized)


def ensure_schedule_bundle() -> Path:
    bundle_path = get_schedule_bundle_path()
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"Runtime schedule bundle not found: {bundle_path}. "
            "Set STHCD_SCHEDULE_BUNDLE to a valid scheduler bundle file."
        )
    return bundle_path


def load_schedule_bundle(bundle_path: Path | None = None) -> Dict[str, Any]:
    bundle_path = bundle_path or ensure_schedule_bundle()
    return torch.load(bundle_path, map_location="cpu")


def _decode_entry(bundle: Dict[str, Any], entry_key: int) -> Dict[str, Any]:
    compressed = bundle["entries"][entry_key]
    return json.loads(zlib.decompress(compressed).decode("utf-8"))


def list_schedule_aliases(bundle_path: Path | None = None) -> Dict[str, str]:
    _ = load_schedule_bundle(bundle_path)
    return dict(PUBLIC_SCHEDULE_ALIASES)


def resolve_schedule_payload(
    schedule_ref: str | Dict[str, Any] | None,
    bundle_path: Path | None = None,
) -> Dict[str, Any]:
    if schedule_ref is None:
        raise ValueError("schedule_ref must not be None.")
    if isinstance(schedule_ref, dict):
        return schedule_ref

    normalized_ref = _normalize_schedule_ref(schedule_ref)
    bundle = load_schedule_bundle(bundle_path)
    entry_key = bundle["aliases"].get(normalized_ref)
    if entry_key is None:
        entry_key = bundle["aliases"].get(schedule_ref)
    if entry_key is None:
        raise KeyError(f"Unknown schedule reference: {schedule_ref}")
    return _decode_entry(bundle, entry_key)
