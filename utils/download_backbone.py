from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils.runtime_paths import ensure_dir, get_runtime_cache_root, resolve_huggingface_cache_paths
from utils.runtime_warnings import suppress_optional_dependency_warnings

DEFAULT_REPO_ID = "STHCD/stable-diffusion-2-1-base"
LOCAL_ENV_PATH = REPO_ROOT / ".env.local"
PROXY_ENV_VARS = (
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "all_proxy",
    "ALL_PROXY",
)

suppress_optional_dependency_warnings()


def _configure_cache_env() -> tuple[Path, Path]:
    cache_root = get_runtime_cache_root()
    hf_home, hub_cache = resolve_huggingface_cache_paths()
    os.environ.setdefault("MPLCONFIGDIR", str(ensure_dir(cache_root / "matplotlib")))
    os.environ.setdefault("TORCH_HOME", str(ensure_dir(cache_root / "torch")))
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hub_cache))
    return Path(os.environ["HF_HOME"]), Path(os.environ["HUGGINGFACE_HUB_CACHE"])


def _unset_proxy_env() -> dict[str, str]:
    removed: dict[str, str] = {}
    for env_name in PROXY_ENV_VARS:
        value = os.environ.pop(env_name, None)
        if value:
            removed[env_name] = value
    return removed


def _load_local_env() -> Path | None:
    if not LOCAL_ENV_PATH.exists():
        return None

    for raw_line in LOCAL_ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            os.environ.setdefault(key, value)

    return LOCAL_ENV_PATH


def _require_download_dependencies():
    try:
        from diffusers import DDPMScheduler
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import GatedRepoError, LocalTokenNotFoundError, RepositoryNotFoundError
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing download dependencies. Please run this script inside the `sthcd` environment "
            "after installing `requirements.txt`."
        ) from exc

    return DDPMScheduler, snapshot_download, RepositoryNotFoundError, GatedRepoError, LocalTokenNotFoundError


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pre-download Stable Diffusion 2.1 assets required by STHCD.",
    )
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face user token. If omitted, the script will use STHCD_HF_TOKEN / HF_TOKEN / HUGGINGFACE_HUB_TOKEN, `.env.local`, or a prior `hf auth login` session.",
    )
    parser.add_argument("--scheduler-only", action="store_true")
    parser.add_argument("--disable-env-proxy", action="store_true")
    return parser


def _resolve_hf_token(explicit_token: str | None) -> str | bool | None:
    if explicit_token:
        return explicit_token
    env_token = (
        os.environ.get("STHCD_HF_TOKEN")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )
    if env_token:
        return env_token
    return None


def _token_source(explicit_token: str | None) -> str:
    if explicit_token:
        return "--token"
    if os.environ.get("STHCD_HF_TOKEN"):
        return "STHCD_HF_TOKEN"
    if os.environ.get("HF_TOKEN"):
        return "HF_TOKEN"
    if os.environ.get("HUGGINGFACE_HUB_TOKEN"):
        return "HUGGINGFACE_HUB_TOKEN"
    return "hf auth login"


def _raise_auth_guidance(repo_id: str, exc: Exception) -> None:
    message = str(exc)
    raise SystemExit(
        "\n".join(
            [
                f"Failed to access Hugging Face repo `{repo_id}`.",
                "This usually means one of the following:",
                "1. The repo id is wrong or the repo is currently unavailable.",
                "2. The repo is private / gated and your machine is not authenticated with a valid Hugging Face token.",
                "",
                "Fix:",
                f"- Open https://huggingface.co/{repo_id} in your browser and confirm the repo exists and is accessible.",
                "- If the repo is private or gated, then run `hf auth login`,",
                "  or pass `--token hf_xxx`,",
                "  or export `STHCD_HF_TOKEN=hf_xxx`,",
                "  or put `STHCD_HF_TOKEN=hf_xxx` into `.env.local`.",
                "",
                f"Original error: {message}",
            ]
        )
    ) from exc


def _download_scheduler_snapshot(
    repo_id: str,
    hub_cache: Path,
    revision: str | None,
    token: str | bool | None,
) -> str:
    _, snapshot_download, RepositoryNotFoundError, GatedRepoError, LocalTokenNotFoundError = _require_download_dependencies()
    try:
        return snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=str(hub_cache),
            allow_patterns=["scheduler/*"],
            token=token,
        )
    except (RepositoryNotFoundError, GatedRepoError, LocalTokenNotFoundError) as exc:
        _raise_auth_guidance(repo_id, exc)
    except Exception as exc:
        if (
            "401" in str(exc)
            or "Unauthorized" in str(exc)
            or "Repository Not Found" in str(exc)
            or "Token is required" in str(exc)
        ):
            _raise_auth_guidance(repo_id, exc)
        raise


def _download_full_snapshot(
    repo_id: str,
    hub_cache: Path,
    revision: str | None,
    token: str | bool | None,
) -> str:
    _, snapshot_download, RepositoryNotFoundError, GatedRepoError, LocalTokenNotFoundError = _require_download_dependencies()
    try:
        return snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=str(hub_cache),
            token=token,
        )
    except (RepositoryNotFoundError, GatedRepoError, LocalTokenNotFoundError) as exc:
        _raise_auth_guidance(repo_id, exc)
    except Exception as exc:
        if (
            "401" in str(exc)
            or "Unauthorized" in str(exc)
            or "Repository Not Found" in str(exc)
            or "Token is required" in str(exc)
        ):
            _raise_auth_guidance(repo_id, exc)
        raise


def _verify_scheduler(repo_id: str, revision: str | None, token: str | bool | None) -> None:
    DDPMScheduler, _, RepositoryNotFoundError, GatedRepoError, LocalTokenNotFoundError = _require_download_dependencies()
    try:
        scheduler = DDPMScheduler.from_pretrained(
            repo_id,
            subfolder="scheduler",
            revision=revision,
            local_files_only=True,
            token=token,
        )
    except (RepositoryNotFoundError, GatedRepoError, LocalTokenNotFoundError) as exc:
        _raise_auth_guidance(repo_id, exc)
    except Exception as exc:
        if (
            "401" in str(exc)
            or "Unauthorized" in str(exc)
            or "Repository Not Found" in str(exc)
            or "Token is required" in str(exc)
        ):
            _raise_auth_guidance(repo_id, exc)
        raise
    print(
        "Verified local scheduler cache: "
        f"num_train_timesteps={scheduler.config.num_train_timesteps}"
    )


def main() -> None:
    args = _build_parser().parse_args()
    loaded_env_path = _load_local_env()
    hf_home, hub_cache = _configure_cache_env()
    token = _resolve_hf_token(args.token)

    if args.disable_env_proxy:
        removed_proxy_env = _unset_proxy_env()
        if removed_proxy_env:
            print("Unset proxy environment variables:")
            for env_name, value in sorted(removed_proxy_env.items()):
                print(f"  {env_name}={value}")

    print(f"HF_HOME={hf_home}")
    print(f"HUGGINGFACE_HUB_CACHE={hub_cache}")
    if loaded_env_path is not None:
        print(f"Loaded local environment overrides from: {loaded_env_path}")
    print(f"Authentication source: {_token_source(args.token)}")

    if args.scheduler_only:
        snapshot_dir = _download_scheduler_snapshot(args.repo_id, hub_cache, args.revision, token)
        _verify_scheduler(args.repo_id, args.revision, token)
        print(f"Scheduler snapshot cached at: {snapshot_dir}")
        return

    snapshot_dir = _download_full_snapshot(args.repo_id, hub_cache, args.revision, token)
    _verify_scheduler(args.repo_id, args.revision, token)
    print(f"Full SD2.1 snapshot cached at: {snapshot_dir}")
    print("The cache is ready for STHCD.")
    print("You can now run experiments with `--local-files-only` if you want strict offline loading.")


if __name__ == "__main__":
    main()
