from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils.runtime_paths import (
    ensure_dir,
    get_repo_root,
    get_results_root,
    get_runtime_cache_root,
    resolve_huggingface_cache_paths,
)
from utils.runtime_warnings import (
    configure_runtime_warning_filters,
    suppress_optional_dependency_warnings,
)

_CACHE_ROOT = get_runtime_cache_root()
_HF_HOME, _HF_HUB_CACHE = resolve_huggingface_cache_paths()
os.environ.setdefault("MPLCONFIGDIR", str(ensure_dir(_CACHE_ROOT / "matplotlib")))
os.environ.setdefault("TORCH_HOME", str(ensure_dir(_CACHE_ROOT / "torch")))
os.environ.setdefault("HF_HOME", str(_HF_HOME))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_HF_HUB_CACHE))

suppress_optional_dependency_warnings()

import lpips
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from utils.experiment_presets import (
    get_modality_defaults,
    normalize_profile_name,
    resolve_rate_preset,
)
from utils.image import BinaryFileDataset, HSIImageDataset, ImageDataset, SARImageDataset
from utils.index_history import (
    build_history_rows,
    save_scheduler_index_history,
    summarize_expected_bitstream,
)
from utils.metrics import (
    calculate_bpp,
    calculate_dists,
    calculate_lpips,
    calculate_psnr,
)
from utils.pipeline_utils import HuggingFaceLoadError, load_residual_hierarchy_scheduler, load_sd21_pipe
from utils.schedule_registry import list_schedule_aliases, resolve_schedule_payload
from utils.seed import set_seed


@dataclass
class ScheduleSelection:
    ref: str | None
    payload: dict[str, Any] | None
    use_progressive_codebook_channel: bool
    use_progressive_shuffle_factor: bool


@dataclass
class PresetSelection:
    profile: str
    rate: str


@dataclass
class ProgressiveReconstructionSelection:
    enabled: bool
    interval: int | None
    stages: list[int]


def _configure_runtime() -> None:
    configure_runtime_warning_filters()
    warnings.simplefilter("ignore")

    if "TRANSFORMERS_CACHE" in os.environ and "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = os.environ["TRANSFORMERS_CACHE"]


def _generic_collate(batch: list[dict[str, Any]]) -> dict[str, list[Any]]:
    keys = batch[0].keys()
    return {key: [item[key] for item in batch] for key in keys}


def _select_dataset_class(modality: str):
    if modality == "visible":
        return ImageDataset
    if modality == "sar":
        return SARImageDataset
    if modality == "hsi":
        return HSIImageDataset
    raise ValueError(f"Unsupported modality: {modality}")


def _apply_modality_defaults(args: argparse.Namespace) -> None:
    defaults = get_modality_defaults(args.modality)

    if args.image_dir is None and args.mode in {"codec", "encode"}:
        args.image_dir = defaults["image_dir"]
    if args.original_image_dir is None and args.mode == "decode":
        args.original_image_dir = defaults["original_image_dir"]
    if args.batch_size is None:
        args.batch_size = defaults["batch_size"]
    if args.num_workers is None:
        args.num_workers = defaults["num_workers"]


def _apply_rate_preset(args: argparse.Namespace) -> PresetSelection | None:
    if args.profile is None and args.rate is None:
        return None

    profile = normalize_profile_name(args.profile or "sthcd")
    rate = args.rate or "high"
    preset = resolve_rate_preset(profile, rate)

    if args.schedule_ref is None:
        args.schedule_ref = preset["schedule_ref"]
    if args.num_timesteps is None:
        args.num_timesteps = preset["num_timesteps"]
    if args.residual_unshuffle_factor is None:
        args.residual_unshuffle_factor = preset["residual_unshuffle_factor"]
    if args.codebook_channel is None:
        args.codebook_channel = preset["codebook_channel"]

    return PresetSelection(profile=profile, rate=rate)


def _resolve_schedule_selection(args: argparse.Namespace) -> ScheduleSelection:
    if args.schedule_ref:
        payload = resolve_schedule_payload(args.schedule_ref)
        return ScheduleSelection(
            ref=args.schedule_ref,
            payload=payload,
            use_progressive_codebook_channel="codebook_channel_map" in payload,
            use_progressive_shuffle_factor="shuffle_factor_map" in payload,
        )

    if args.channel_schedule_path and args.shuffle_schedule_path:
        channel_payload = resolve_schedule_payload(args.channel_schedule_path)
        shuffle_payload = resolve_schedule_payload(args.shuffle_schedule_path)
        merged_payload: dict[str, Any] = {}
        merged_payload.update(channel_payload)
        merged_payload.update(shuffle_payload)

        if "cosine_max_beta_map" not in merged_payload:
            if "cosine_max_beta_map" in shuffle_payload:
                merged_payload["cosine_max_beta_map"] = shuffle_payload["cosine_max_beta_map"]
            elif "cosine_max_beta_map" in channel_payload:
                merged_payload["cosine_max_beta_map"] = channel_payload["cosine_max_beta_map"]
            else:
                merged_payload["cosine_max_beta_map"] = resolve_schedule_payload("beta")

        return ScheduleSelection(
            ref=f"{args.shuffle_schedule_path}+{args.channel_schedule_path}",
            payload=merged_payload,
            use_progressive_codebook_channel=True,
            use_progressive_shuffle_factor=True,
        )

    if args.shuffle_schedule_path:
        payload = resolve_schedule_payload(args.shuffle_schedule_path)
        return ScheduleSelection(
            ref=args.shuffle_schedule_path,
            payload=payload,
            use_progressive_codebook_channel="codebook_channel_map" in payload,
            use_progressive_shuffle_factor="shuffle_factor_map" in payload,
        )

    if args.channel_schedule_path:
        payload = resolve_schedule_payload(args.channel_schedule_path)
        return ScheduleSelection(
            ref=args.channel_schedule_path,
            payload=payload,
            use_progressive_codebook_channel="codebook_channel_map" in payload,
            use_progressive_shuffle_factor="shuffle_factor_map" in payload,
        )

    return ScheduleSelection(
        ref=None,
        payload=None,
        use_progressive_codebook_channel=False,
        use_progressive_shuffle_factor=False,
    )


def _resolve_experiment_root(args: argparse.Namespace, experiment_name: str) -> Path:
    if args.experiment_dir:
        base_dir = Path(args.experiment_dir)
        if not base_dir.is_absolute():
            base_dir = get_repo_root() / base_dir
    else:
        base_dir = get_results_root() / args.modality
    return ensure_dir(base_dir / experiment_name)


def _normalize_image_tensors(image_tensors: list[torch.Tensor], device: str) -> torch.Tensor:
    processed = []
    for tensor in image_tensors:
        if tensor.dim() == 4 and tensor.size(0) == 1:
            tensor = tensor.squeeze(0)
        processed.append(tensor)

    images = torch.stack(processed).to(device)
    if images.min() < 0:
        images = (images + 1.0) / 2.0
    return torch.clamp(images, 0.0, 1.0)


def _output_item_at(output: Any, index: int) -> Any:
    if hasattr(output, "images"):
        images = output.images
        if isinstance(images, list):
            return images[index]
        return images[index]

    if isinstance(output, (list, tuple)):
        return output[index]

    return output


def _save_output_item(item: Any, output_dir: Path, image_id: str) -> Image.Image:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{image_id}.png"

    if hasattr(item, "save"):
        pil_image = item
        pil_image.save(png_path)
        return pil_image

    if isinstance(item, torch.Tensor):
        tensor = item.detach().cpu()
        if tensor.dim() == 4 and tensor.size(0) == 1:
            tensor = tensor.squeeze(0)
        if tensor.min() < 0:
            tensor = (tensor + 1.0) / 2.0
        tensor = torch.clamp(tensor, 0.0, 1.0)
        if tensor.dim() == 3 and tensor.shape[0] in {1, 3}:
            array = tensor.permute(1, 2, 0).numpy()
        else:
            array = tensor.numpy()
    else:
        array = np.asarray(item)

    if array.dtype != np.uint8:
        if np.max(array) <= 1.0:
            array = np.clip(array, 0.0, 1.0)
            array = (array * 255.0).astype(np.uint8)
        else:
            array = array.astype(np.uint8)

    if array.ndim == 3 and array.shape[0] in {1, 3}:
        array = np.transpose(array, (1, 2, 0))
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]

    pil_image = Image.fromarray(array)
    pil_image.save(png_path)
    return pil_image

def _compute_schedule_bpp(
    schedule_selection: ScheduleSelection,
    args: argparse.Namespace,
    image_size: int,
) -> float:
    return calculate_bpp(
        T=args.num_timesteps,
        K=args.codebook_size,
        H=image_size,
        W=image_size,
        residual_unshuffle_factor=args.residual_unshuffle_factor,
        codebook_channel=args.codebook_channel,
        schedule_payload=schedule_selection.payload,
    )


def _parse_progressive_reconstruction_stages(
    raw_value: str | None,
    num_timesteps: int,
) -> list[int]:
    if raw_value in (None, ""):
        return []

    resolved: set[int] = set()
    for raw_item in str(raw_value).split(","):
        item = raw_item.strip()
        if not item:
            continue
        stage = int(item)
        if stage < 1 or stage > int(num_timesteps):
            raise ValueError(
                f"--progressive_reconstruction_stages must stay within [1, {num_timesteps}], got {stage}."
            )
        resolved.add(stage)
    return sorted(resolved)


def _resolve_progressive_reconstruction_selection(
    args: argparse.Namespace,
) -> ProgressiveReconstructionSelection:
    legacy_enabled = bool(getattr(args, "save_intermediates", False))
    enabled = bool(args.save_progressive_reconstruction or legacy_enabled)

    interval = args.progressive_reconstruction_interval
    if interval is not None and interval <= 0:
        raise ValueError("--progressive_reconstruction_interval must be positive.")

    stages = _parse_progressive_reconstruction_stages(
        args.progressive_reconstruction_stages,
        args.num_timesteps,
    )

    if not enabled:
        interval = None
        stages = []

    return ProgressiveReconstructionSelection(
        enabled=enabled,
        interval=interval,
        stages=stages,
    )


def _collect_progressive_reconstruction_rows(pipe) -> list[dict[str, Any]]:
    records = getattr(pipe, "progressive_reconstruction_records", None)
    if not records:
        return []
    return [dict(record) for record in records]


def _write_progressive_reconstruction_manifest(
    experiment_root: Path,
    progressive_rows: list[dict[str, Any]],
) -> None:
    if not progressive_rows:
        return
    reports_dir = ensure_dir(experiment_root / "reports")
    manifest_df = pd.DataFrame(progressive_rows)
    sort_columns = [column for column in ("image_id", "stage", "mode", "timestep") if column in manifest_df.columns]
    if sort_columns:
        manifest_df = manifest_df.sort_values(sort_columns).reset_index(drop=True)
    manifest_df.to_csv(
        reports_dir / "progressive_reconstruction_manifest.csv",
        index=False,
    )


def _save_config_snapshot(experiment_root: Path, args: argparse.Namespace, resolved: dict[str, Any]) -> None:
    reports_dir = ensure_dir(experiment_root / "reports")
    snapshot = {
        "mode": args.mode,
        "modality": args.modality,
        "image_dir": args.image_dir,
        "original_image_dir": args.original_image_dir,
        "bin_dir": args.bin_dir,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "image_size": args.image_size,
        "codebook_size": args.codebook_size,
        "num_timesteps": args.num_timesteps,
        "residual_factor": args.residual_unshuffle_factor,
        "codebook_channel": args.codebook_channel,
        "schedule_ref": args.schedule_ref,
        "channel_schedule_path": args.channel_schedule_path,
        "residual_factor_schedule_path": args.shuffle_schedule_path,
        "profile": args.profile,
        "rate": args.rate,
        "experiment_dir": args.experiment_dir,
        "experiment_name": args.experiment_name,
        "device": args.device,
        "data_type": args.data_type,
        "seed": args.seed,
        "local_files_only": args.local_files_only,
        "save_progressive_reconstruction": args.save_progressive_reconstruction,
        "progressive_reconstruction_interval": args.progressive_reconstruction_interval,
        "progressive_reconstruction_stages": args.progressive_reconstruction_stages,
        "enable_dists": args.enable_dists,
    }
    snapshot.update(resolved)
    (reports_dir / "run_config.json").write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _stat_bitstream_file(bin_dir: str | Path, image_id: str, image_size: int | tuple[int, int]) -> dict[str, Any]:
    bin_path = Path(bin_dir) / f"{image_id}.bin"
    file_size_bytes = bin_path.stat().st_size
    file_size_bits = file_size_bytes * 8
    image_area = image_size * image_size if isinstance(image_size, int) else int(image_size[0]) * int(image_size[1])
    return {
        "image_id": image_id,
        "file_size_bytes": file_size_bytes,
        "file_size_bits": file_size_bits,
        "actual_bpp": file_size_bits / float(image_area),
    }


def _run_codec_mode(
    pipe,
    dataloader: DataLoader,
    args: argparse.Namespace,
    experiment_root: Path,
    generator,
    lpips_fn,
    enable_dists: bool,
    schedule_bpp: float,
    progressive_selection: ProgressiveReconstructionSelection,
) -> None:
    recon_dir = ensure_dir(experiment_root / "reconstructions")
    progressive_dir = (
        ensure_dir(experiment_root / "progressive_reconstruction")
        if progressive_selection.enabled
        else None
    )
    reports_dir = ensure_dir(experiment_root / "reports")

    all_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    progressive_rows: list[dict[str, Any]] = []

    start_time = time.time()
    for batch_data in dataloader:
        input_images = _normalize_image_tensors(batch_data["image"], pipe.device)

        with torch.no_grad():
            output = pipe(
                prompt=[""] * len(batch_data["image_id"]),
                image=input_images,
                num_inference_steps=args.num_timesteps,
                height=args.image_size,
                width=args.image_size,
                guidance_scale=0.0,
                eta=1.0,
                strength=1.0,
                generator=generator,
                mode="codec",
                save_progressive_reconstruction=progressive_selection.enabled,
                progressive_reconstruction_dir=str(progressive_dir) if progressive_dir else None,
                progressive_reconstruction_interval=progressive_selection.interval,
                progressive_reconstruction_stages=progressive_selection.stages,
                batch_id=batch_data["image_id"],
                return_step_metrics=False,
            )

        progressive_rows.extend(_collect_progressive_reconstruction_rows(pipe))

        history_rows.extend(
            build_history_rows(
                history=pipe.scheduler.index_history,
                num_timesteps=args.num_timesteps,
                batch_size=len(batch_data["image_id"]),
                batch_ids=batch_data["image_id"],
            )
        )

        for item_idx, image_id in enumerate(batch_data["image_id"]):
            item = _output_item_at(output, item_idx)
            decoded_image = _save_output_item(item, recon_dir, image_id)
            original_path = batch_data["image_path"][item_idx]

            row = {
                "image_id": image_id,
                "psnr": calculate_psnr(original_path, decoded_image, data_range=255),
                "lpips": (
                    calculate_lpips(original_path, decoded_image, lpips_fn, pipe.device)
                    if lpips_fn is not None
                    else np.nan
                ),
                "bpp": schedule_bpp,
            }
            if enable_dists:
                row["dists"] = calculate_dists(original_path, decoded_image, pipe.device)

            all_rows.append(row)

    elapsed = time.time() - start_time

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(reports_dir / "reconstruction_results_per_image.csv", index=False)
    pd.DataFrame(history_rows).to_csv(reports_dir / "index_history.csv", index=False)
    _write_progressive_reconstruction_manifest(experiment_root, progressive_rows)

    summary_lines = [
        f"Mode: codec",
        f"Modality: {args.modality}",
        f"Images: {len(results_df)}",
        f"Elapsed Seconds: {elapsed:.3f}",
        f"BPP: {schedule_bpp:.6f}",
        f"Average PSNR: {results_df['psnr'].mean():.4f}",
    ]
    if "lpips" in results_df and results_df["lpips"].notna().any():
        summary_lines.append(f"Average LPIPS: {results_df['lpips'].dropna().mean():.6f}")
    if enable_dists and "dists" in results_df:
        summary_lines.append(f"Average DISTS: {results_df['dists'].mean():.6f}")
    (reports_dir / "reconstruction_metrics.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def _run_encode_mode(
    pipe,
    dataloader: DataLoader,
    args: argparse.Namespace,
    experiment_root: Path,
    generator,
    schedule_selection: ScheduleSelection,
    schedule_bpp: float,
    progressive_selection: ProgressiveReconstructionSelection,
) -> None:
    bins_dir = ensure_dir(experiment_root / "bins")
    progressive_dir = (
        ensure_dir(experiment_root / "progressive_reconstruction")
        if progressive_selection.enabled
        else None
    )
    reports_dir = ensure_dir(experiment_root / "reports")

    history_rows: list[dict[str, Any]] = []
    bitstream_rows: list[dict[str, Any]] = []
    progressive_rows: list[dict[str, Any]] = []

    for batch_data in dataloader:
        input_images = _normalize_image_tensors(batch_data["image"], pipe.device)

        with torch.no_grad():
            pipe(
                prompt=[""] * len(batch_data["image_id"]),
                image=input_images,
                num_inference_steps=args.num_timesteps,
                height=args.image_size,
                width=args.image_size,
                guidance_scale=0.0,
                eta=1.0,
                strength=1.0,
                generator=generator,
                mode="encode",
                save_progressive_reconstruction=progressive_selection.enabled,
                progressive_reconstruction_dir=str(progressive_dir) if progressive_dir else None,
                progressive_reconstruction_interval=progressive_selection.interval,
                progressive_reconstruction_stages=progressive_selection.stages,
                batch_id=batch_data["image_id"],
                output_type="latent",
            )

        progressive_rows.extend(_collect_progressive_reconstruction_rows(pipe))

        bitstream_rows.extend(
            save_scheduler_index_history(
            bin_dir=bins_dir,
            batch_ids=batch_data["image_id"],
            history=pipe.scheduler.index_history,
            num_timesteps=args.num_timesteps,
            batch_size=len(batch_data["image_id"]),
            codebook_size=args.codebook_size,
            image_size=args.image_size,
            metadata={
                "modality": args.modality,
                "schedule_ref": schedule_selection.ref,
                "num_timesteps": args.num_timesteps,
                "codebook_size": args.codebook_size,
            },
        )
        )

        history_rows.extend(
            build_history_rows(
                history=pipe.scheduler.index_history,
                num_timesteps=args.num_timesteps,
                batch_size=len(batch_data["image_id"]),
                batch_ids=batch_data["image_id"],
            )
        )

    pd.DataFrame(history_rows).to_csv(reports_dir / "index_history.csv", index=False)
    bitstream_df = pd.DataFrame(bitstream_rows)
    bitstream_df.to_csv(reports_dir / "encoded_bitstreams.csv", index=False)
    _write_progressive_reconstruction_manifest(experiment_root, progressive_rows)
    saved_bpp = bitstream_df["actual_bpp"].mean() if not bitstream_df.empty else float("nan")
    file_size_bytes = bitstream_df["file_size_bytes"].mean() if not bitstream_df.empty else float("nan")
    summary_bin_dir = os.path.relpath(bins_dir, get_repo_root())
    (reports_dir / "encoding_summary.txt").write_text(
        "\n".join(
            [
                "Mode: encode",
                f"Modality: {args.modality}",
                f"Bin Directory: {summary_bin_dir}",
                f"Schedule Ref: {schedule_selection.ref}",
                f"Expected BPP: {schedule_bpp:.6f}",
                f"Saved BPP: {saved_bpp:.6f}",
                f"Average File Size Bytes: {file_size_bytes:.3f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _run_decode_mode(
    pipe,
    dataloader: DataLoader,
    args: argparse.Namespace,
    experiment_root: Path,
    generator,
    lpips_fn,
    enable_dists: bool,
    schedule_bpp: float,
    progressive_selection: ProgressiveReconstructionSelection,
) -> None:
    decoded_dir = ensure_dir(experiment_root / "decoded")
    progressive_dir = (
        ensure_dir(experiment_root / "progressive_reconstruction")
        if progressive_selection.enabled
        else None
    )
    reports_dir = ensure_dir(experiment_root / "reports")

    all_rows: list[dict[str, Any]] = []
    bitstream_rows: list[dict[str, Any]] = []
    progressive_rows: list[dict[str, Any]] = []

    for batch_data in dataloader:
        with torch.no_grad():
            output = pipe(
                prompt=[""] * len(batch_data["image_id"]),
                num_inference_steps=args.num_timesteps,
                height=args.image_size,
                width=args.image_size,
                guidance_scale=0.0,
                eta=1.0,
                strength=1.0,
                generator=generator,
                mode="decode",
                save_progressive_reconstruction=progressive_selection.enabled,
                progressive_reconstruction_dir=str(progressive_dir) if progressive_dir else None,
                progressive_reconstruction_interval=progressive_selection.interval,
                progressive_reconstruction_stages=progressive_selection.stages,
                batch_id=batch_data["image_id"],
                bin_dir=args.bin_dir,
                return_step_metrics=False,
            )

        progressive_rows.extend(_collect_progressive_reconstruction_rows(pipe))

        for item_idx, image_id in enumerate(batch_data["image_id"]):
            item = _output_item_at(output, item_idx)
            decoded_image = _save_output_item(item, decoded_dir, image_id)
            original_path = batch_data["original_image_path"][item_idx]
            bitstream_stats = _stat_bitstream_file(args.bin_dir, image_id, args.image_size)
            bitstream_rows.append(bitstream_stats)

            row = {"image_id": image_id}
            row["bpp"] = bitstream_stats["actual_bpp"]
            row["file_size_bytes"] = bitstream_stats["file_size_bytes"]
            if original_path is not None:
                row["psnr"] = calculate_psnr(original_path, decoded_image, data_range=255)
                if lpips_fn is not None:
                    row["lpips"] = calculate_lpips(original_path, decoded_image, lpips_fn, pipe.device)
                if enable_dists:
                    row["dists"] = calculate_dists(original_path, decoded_image, pipe.device)

            all_rows.append(row)

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(reports_dir / "decoded_results_per_image.csv", index=False)
    pd.DataFrame(bitstream_rows).to_csv(reports_dir / "decoded_bitstreams.csv", index=False)
    _write_progressive_reconstruction_manifest(experiment_root, progressive_rows)
    saved_bpp = results_df["bpp"].mean() if not results_df.empty else float("nan")

    summary_lines = [
        f"Mode: decode",
        f"Modality: {args.modality}",
        f"Images: {len(results_df)}",
        f"Expected BPP: {schedule_bpp:.6f}",
        f"Saved BPP: {saved_bpp:.6f}",
    ]
    if "psnr" in results_df:
        summary_lines.append(f"Average PSNR: {results_df['psnr'].mean():.4f}")
    if "lpips" in results_df and results_df["lpips"].notna().any():
        summary_lines.append(f"Average LPIPS: {results_df['lpips'].dropna().mean():.6f}")
    if enable_dists and "dists" in results_df:
        summary_lines.append(f"Average DISTS: {results_df['dists'].mean():.6f}")

    (reports_dir / "decode_metrics.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def _build_parser(default_modality: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run STHCD experiments.")
    parser.add_argument("--mode", type=str, default="codec", choices=["codec", "encode", "decode"])
    parser.add_argument(
        "--modality",
        type=str,
        default=default_modality or "visible",
        choices=["visible", "sar", "hsi"],
    )
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--original_image_dir", type=str, default=None)
    parser.add_argument("--bin_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--codebook_size", type=int, default=256)
    parser.add_argument("--num_timesteps", type=int, default=None)
    parser.add_argument(
        "--residual_factor",
        dest="residual_unshuffle_factor",
        type=int,
        default=None,
        metavar="RESIDUAL_FACTOR",
    )
    parser.add_argument(
        "--residual_unshuffle_factor",
        dest="residual_unshuffle_factor",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--codebook_channel", type=int, default=None)
    parser.add_argument(
        "--schedule_ref",
        type=str,
        default=None,
        help="Schedule alias. Recommended public aliases: spatial, channel, sthcd.",
    )
    parser.add_argument("--channel_schedule_path", type=str, default=None)
    parser.add_argument(
        "--residual_factor_schedule_path",
        dest="shuffle_schedule_path",
        type=str,
        default=None,
        metavar="RESIDUAL_FACTOR_SCHEDULE_PATH",
    )
    parser.add_argument("--shuffle_schedule_path", dest="shuffle_schedule_path", type=str, help=argparse.SUPPRESS)
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help=(
            "Bitrate profile family. Recommended public names: baseline, spatial, channel, sthcd. "
            "Legacy aliases shuffle and channel_shuffle remain accepted."
        ),
    )
    parser.add_argument("--rate", type=str, default=None, choices=["low", "mid", "high"])
    parser.add_argument("--experiment_dir", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--data_type", type=str, default=None, choices=["float16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--save_progressive_reconstruction", action="store_true")
    parser.add_argument("--progressive_reconstruction_interval", type=int, default=None)
    parser.add_argument("--progressive_reconstruction_stages", type=str, default=None)
    parser.add_argument("--save_intermediates", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--enable_dists", dest="enable_dists", action="store_true")
    parser.add_argument("--disable_dists", dest="enable_dists", action="store_false")
    parser.set_defaults(enable_dists=None)
    return parser


def main(default_modality: str | None = None) -> None:
    _configure_runtime()
    parser = _build_parser(default_modality=default_modality)
    args = parser.parse_args()

    preset_selection = _apply_rate_preset(args)
    _apply_modality_defaults(args)
    if args.num_timesteps is None:
        args.num_timesteps = 100
    if args.residual_unshuffle_factor is None:
        args.residual_unshuffle_factor = 1
    if args.codebook_channel is None:
        args.codebook_channel = 8
    progressive_selection = _resolve_progressive_reconstruction_selection(args)
    args.save_progressive_reconstruction = progressive_selection.enabled
    args.progressive_reconstruction_interval = progressive_selection.interval
    args.progressive_reconstruction_stages = progressive_selection.stages

    if args.mode in {"codec", "encode"} and not args.image_dir:
        raise ValueError("--image_dir is required for codec and encode modes.")
    if args.mode == "decode" and not args.bin_dir:
        raise ValueError("--bin_dir is required for decode mode.")
    if args.mode == "decode" and not args.original_image_dir:
        raise ValueError("--original_image_dir is required for decode mode.")

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    data_type = args.data_type or "float16"
    enable_dists = args.enable_dists
    if enable_dists is None:
        enable_dists = args.modality in {"sar", "hsi"}

    schedule_selection = _resolve_schedule_selection(args)
    generator = set_seed(args.seed, "cpu")

    schedule_suffix = "base"
    if preset_selection is not None:
        schedule_suffix = f"{preset_selection.profile}_{preset_selection.rate}"
    elif schedule_selection.ref:
        schedule_suffix = Path(schedule_selection.ref.split("+")[0]).stem

    experiment_name = args.experiment_name or (
        f"{args.mode}_{args.modality}_K{args.codebook_size}_T{args.num_timesteps}_"
        f"C{args.codebook_channel}_F{args.residual_unshuffle_factor}_{schedule_suffix}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    experiment_root = _resolve_experiment_root(args, experiment_name)

    if args.mode == "decode":
        dataset = BinaryFileDataset(
            bin_dir=args.bin_dir,
            original_image_dir=args.original_image_dir,
            image_size=args.image_size,
        )
    else:
        dataset_cls = _select_dataset_class(args.modality)
        dataset = dataset_cls(
            args.image_dir,
            image_size=args.image_size,
        )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=_generic_collate,
        num_workers=args.num_workers,
    )

    schedule_payload = schedule_selection.payload
    try:
        scheduler = load_residual_hierarchy_scheduler(
            num_timesteps=args.num_timesteps,
            codebook_size=args.codebook_size,
            codebook_dims=(
                args.codebook_channel,
                args.image_size // (8 * args.residual_unshuffle_factor),
                args.image_size // (8 * args.residual_unshuffle_factor),
            ),
            data_type=data_type,
            residual_unshuffle_factor=args.residual_unshuffle_factor,
            use_progressive_codebook_channel=schedule_selection.use_progressive_codebook_channel,
            channel_schedule_path=schedule_selection.ref if schedule_selection.use_progressive_codebook_channel else None,
            use_progressive_shuffle_factor=schedule_selection.use_progressive_shuffle_factor,
            shuffle_schedule_path=schedule_selection.ref if schedule_selection.use_progressive_shuffle_factor else None,
            channel_schedule_data=schedule_payload if schedule_selection.use_progressive_codebook_channel else None,
            shuffle_schedule_data=schedule_payload if schedule_selection.use_progressive_shuffle_factor else None,
            local_files_only=args.local_files_only,
        )
        scheduler.set_timesteps(args.num_timesteps, device="cpu")
        bitstream_summary = summarize_expected_bitstream(
            scheduler=scheduler,
            timesteps=scheduler.timesteps,
            codebook_size=args.codebook_size,
            image_size=args.image_size,
        )
        schedule_bpp = float(bitstream_summary["bpp"])
        pipe = load_sd21_pipe(
            scheduler,
            device,
            data_type,
            local_files_only=args.local_files_only,
        )
    except HuggingFaceLoadError as exc:
        raise SystemExit(f"{exc}\n") from exc

    _save_config_snapshot(
        experiment_root,
        args,
        {
            "resolved_device": device,
            "resolved_data_type": data_type,
            "resolved_schedule_ref": schedule_selection.ref,
            "resolved_schedule_bpp": schedule_bpp,
            "resolved_schedule_bits_per_symbol": int(bitstream_summary["bits_per_symbol"]),
            "resolved_schedule_total_symbols": int(bitstream_summary["total_symbols"]),
            "resolved_schedule_total_bits": int(bitstream_summary["total_bits"]),
            "resolved_schedule_total_bytes": int(bitstream_summary["total_bytes"]),
            "resolved_profile": preset_selection.profile if preset_selection else None,
            "resolved_rate": preset_selection.rate if preset_selection else None,
            "resolved_progressive_reconstruction": progressive_selection.enabled,
            "resolved_progressive_reconstruction_interval": progressive_selection.interval,
            "resolved_progressive_reconstruction_stages": progressive_selection.stages,
            "enable_dists": enable_dists,
        },
    )

    lpips_fn = None
    if args.mode in {"codec", "decode"}:
        try:
            lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device)
        except Exception as exc:
            print(f"Warning: LPIPS initialization failed and will be skipped: {exc}")

    if args.mode == "codec":
        _run_codec_mode(
            pipe,
            dataloader,
            args,
            experiment_root,
            generator,
            lpips_fn,
            enable_dists,
            schedule_bpp,
            progressive_selection,
        )
    elif args.mode == "encode":
        _run_encode_mode(
            pipe,
            dataloader,
            args,
            experiment_root,
            generator,
            schedule_selection,
            schedule_bpp,
            progressive_selection,
        )
    else:
        _run_decode_mode(
            pipe,
            dataloader,
            args,
            experiment_root,
            generator,
            lpips_fn,
            enable_dists,
            schedule_bpp,
            progressive_selection,
        )

    summary_root = os.path.relpath(experiment_root, get_repo_root())
    print(f"Results saved to: {summary_root}")


def main_list_schedules() -> None:
    aliases = list_schedule_aliases()
    for alias in sorted(aliases):
        print(alias)
