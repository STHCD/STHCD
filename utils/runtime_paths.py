from __future__ import annotations

import os
from pathlib import Path


def get_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _as_path(value: str | os.PathLike[str] | None, fallback: Path) -> Path:
    if value is None:
        return fallback
    path = Path(value)
    return path if path.is_absolute() else (get_repo_root() / path)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_outputs_root() -> Path:
    return ensure_dir(
        _as_path(os.environ.get("STHCD_OUTPUTS_DIR"), get_repo_root() / "outputs")
    )


def get_resources_root() -> Path:
    return ensure_dir(get_repo_root() / "resources")


def get_runtime_cache_root() -> Path:
    return ensure_dir(get_repo_root() / ".cache")


def get_results_root() -> Path:
    return ensure_dir(get_outputs_root() / "results")


def get_codebook_root() -> Path:
    explicit = os.environ.get("STHCD_CODEBOOK_DIR")
    if explicit is not None:
        return ensure_dir(_as_path(explicit, get_resources_root() / "codebook"))

    preferred = get_resources_root() / "codebook"
    legacy = get_repo_root() / "codebook"
    if preferred.exists() or not legacy.exists():
        return ensure_dir(preferred)
    return ensure_dir(legacy)


def get_schedule_bundle_path() -> Path:
    explicit = os.environ.get("STHCD_SCHEDULE_BUNDLE")
    if explicit is not None:
        return _as_path(explicit, get_repo_root() / "schedulers" / ".core.ckpt")

    return get_repo_root() / "schedulers" / ".core.ckpt"


def build_codebook_cache_dir(data_type: str, variant: str, cache_name: str) -> str:
    cache_root = ensure_dir(get_codebook_root() / variant)
    return str(cache_root / cache_name)


def resolve_codebook_cache_dir(
    data_type: str,
    variant: str,
    cache_name: str,
    *,
    legacy_variants: tuple[str, ...] = (),
    legacy_cache_names: tuple[str, ...] = (),
) -> str:
    preferred = get_codebook_root() / variant / cache_name
    candidates = [preferred]

    legacy_roots = [
        get_codebook_root() / f"{data_type}_{variant}",
        get_codebook_root() / f"float16_{variant}",
        get_codebook_root() / f"float32_{variant}",
    ]
    for legacy_root in legacy_roots:
        candidates.append(legacy_root / cache_name)

    for legacy_variant in legacy_variants:
        candidates.append(get_codebook_root() / legacy_variant / cache_name)
        for legacy_cache_name in legacy_cache_names:
            candidates.append(get_codebook_root() / legacy_variant / legacy_cache_name)
        for legacy_root in (
            get_codebook_root() / f"{data_type}_{legacy_variant}",
            get_codebook_root() / f"float16_{legacy_variant}",
            get_codebook_root() / f"float32_{legacy_variant}",
        ):
            candidates.append(legacy_root / cache_name)
            for legacy_cache_name in legacy_cache_names:
                candidates.append(legacy_root / legacy_cache_name)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    preferred.parent.mkdir(parents=True, exist_ok=True)
    return str(preferred)


def resolve_huggingface_cache_paths() -> tuple[Path, Path]:
    explicit_hf_home = os.environ.get("HF_HOME")
    explicit_hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")

    if explicit_hf_home:
        hf_home = _as_path(explicit_hf_home, get_runtime_cache_root() / "huggingface")
        hub_cache = (
            _as_path(explicit_hub_cache, hf_home / "hub")
            if explicit_hub_cache
            else hf_home / "hub"
        )
        return ensure_dir(hf_home), ensure_dir(hub_cache)

    if explicit_hub_cache:
        hub_cache = _as_path(explicit_hub_cache, get_runtime_cache_root() / "huggingface" / "hub")
        return ensure_dir(hub_cache.parent), ensure_dir(hub_cache)

    repo_hf_home = ensure_dir(get_runtime_cache_root() / "huggingface")
    repo_hub_cache = ensure_dir(repo_hf_home / "hub")
    return repo_hf_home, repo_hub_cache


def resolve_results_dir(experiment_dir: str | None = None) -> Path:
    if not experiment_dir:
        return get_results_root()

    experiment_path = Path(experiment_dir)
    if experiment_path.is_absolute():
        return ensure_dir(experiment_path)

    return ensure_dir(get_repo_root() / experiment_path)
