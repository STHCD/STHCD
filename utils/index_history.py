from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from utils.residual_decode import build_sliding_windows


SD21_LATENT_CHANNELS = 4


def to_long_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().long()
    return torch.as_tensor(value, dtype=torch.long)


def serialize_history_step(step: Any) -> dict[str, Any]:
    tensor = to_long_tensor(step)
    return {
        "shape": list(tensor.shape),
        "values": tensor.reshape(-1).tolist(),
    }


def deserialize_history_step(payload: Any) -> torch.Tensor:
    if isinstance(payload, dict) and "shape" in payload and "values" in payload:
        tensor = torch.tensor(payload["values"], dtype=torch.long)
        shape = payload["shape"]
        return tensor.reshape(shape) if shape else tensor.reshape([])
    return to_long_tensor(payload)


def get_fixed_width_bits(codebook_size: int) -> int:
    if codebook_size <= 1:
        raise ValueError(f"codebook_size must be greater than 1, got {codebook_size}.")

    bits = math.log2(codebook_size)
    rounded_bits = int(round(bits))
    if not math.isclose(bits, rounded_bits, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "Raw fixed-width bitstreams require codebook_size to be a power of two so that log2(K) is integral. "
            f"Got K={codebook_size}."
        )
    return rounded_bits


def _resolve_image_area(image_size: int | tuple[int, int]) -> int:
    if isinstance(image_size, int):
        if image_size <= 0:
            raise ValueError(f"image_size must be positive, got {image_size}.")
        return image_size * image_size

    height, width = int(image_size[0]), int(image_size[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}.")
    return height * width


def _step_index_from_timestep(scheduler: Any, timestep: Any) -> int:
    t_value = timestep.item() if isinstance(timestep, torch.Tensor) else int(timestep)
    return int(t_value * scheduler.num_inference_steps / 1000)


def _step_codebook_channels(scheduler: Any, step_index: int) -> int:
    if hasattr(scheduler, "_get_current_channel_config"):
        return int(scheduler._get_current_channel_config(step_index))

    codebook_dims = getattr(scheduler, "codebook_dims", None)
    if not codebook_dims:
        raise AttributeError("Scheduler does not expose codebook_dims, so codebook channel counts cannot be inferred.")
    return int(codebook_dims[0])


def _step_residual_factor(scheduler: Any, step_index: int) -> int:
    if hasattr(scheduler, "_get_current_shuffle_config"):
        return int(scheduler._get_current_shuffle_config(step_index))

    if not hasattr(scheduler, "residual_unshuffle_factor"):
        raise AttributeError(
            "Scheduler does not expose residual_unshuffle_factor, so residual factor cannot be inferred."
        )
    return int(scheduler.residual_unshuffle_factor)


def infer_step_symbol_counts(
    scheduler: Any,
    timesteps: Iterable[Any],
) -> list[int]:
    counts: list[int] = []

    for timestep in timesteps:
        step_index = _step_index_from_timestep(scheduler, timestep)
        residual_factor = _step_residual_factor(scheduler, step_index)
        codebook_channels = _step_codebook_channels(scheduler, step_index)
        transformed_channels = SD21_LATENT_CHANNELS * (residual_factor**2)

        if transformed_channels < codebook_channels:
            raise ValueError(
                f"Channel mismatch at step_index={step_index}: transformed_channels={transformed_channels}, "
                f"codebook_channels={codebook_channels}."
            )

        if transformed_channels % codebook_channels == 0:
            counts.append(transformed_channels // codebook_channels)
            continue

        windows = build_sliding_windows(
            total_channels=transformed_channels,
            codebook_channels=codebook_channels,
            device=torch.device("cpu"),
            stride=max(int(getattr(scheduler, "slide_stride", 1)), 1),
            prefer_ch0_when_sliding=bool(getattr(scheduler, "prefer_ch0_when_sliding", True)),
        )
        counts.append(len(windows))

    return counts


def summarize_expected_bitstream(
    scheduler: Any,
    timesteps: Iterable[Any],
    codebook_size: int,
    image_size: int | tuple[int, int],
) -> dict[str, float | int]:
    bits_per_symbol = get_fixed_width_bits(codebook_size)
    step_symbol_counts = infer_step_symbol_counts(scheduler, timesteps)
    total_symbols = int(sum(step_symbol_counts))
    total_bits = total_symbols * bits_per_symbol

    if total_bits % 8 != 0:
        raise ValueError(
            "The raw fixed-width bitstream is not byte-aligned, so the saved file size cannot exactly match the "
            f"theoretical BPP. total_bits={total_bits}."
        )

    image_area = _resolve_image_area(image_size)
    return {
        "bits_per_symbol": bits_per_symbol,
        "step_symbol_counts": step_symbol_counts,
        "total_symbols": total_symbols,
        "total_bits": total_bits,
        "total_bytes": total_bits // 8,
        "bpp": total_bits / float(image_area),
    }


def _pack_symbols(symbols: list[int], bits_per_symbol: int) -> bytes:
    if not symbols:
        return b""

    max_symbol = (1 << bits_per_symbol) - 1
    buffer = 0
    buffered_bits = 0
    payload = bytearray()

    for symbol in symbols:
        symbol = int(symbol)
        if symbol < 0 or symbol > max_symbol:
            raise ValueError(f"Symbol {symbol} is outside the fixed-width range [0, {max_symbol}].")

        buffer = (buffer << bits_per_symbol) | symbol
        buffered_bits += bits_per_symbol

        while buffered_bits >= 8:
            shift = buffered_bits - 8
            payload.append((buffer >> shift) & 0xFF)
            buffer &= (1 << shift) - 1 if shift > 0 else 0
            buffered_bits -= 8

    if buffered_bits != 0:
        raise ValueError(
            "The raw fixed-width bitstream is not byte-aligned. "
            f"Remaining buffered_bits={buffered_bits}."
        )

    return bytes(payload)


def _unpack_symbols(payload: bytes, bits_per_symbol: int, symbol_count: int) -> list[int]:
    if symbol_count == 0:
        return []

    values: list[int] = []
    buffer = 0
    buffered_bits = 0
    mask = (1 << bits_per_symbol) - 1

    for byte in payload:
        buffer = (buffer << 8) | byte
        buffered_bits += 8

        while buffered_bits >= bits_per_symbol and len(values) < symbol_count:
            shift = buffered_bits - bits_per_symbol
            values.append((buffer >> shift) & mask)
            buffer &= (1 << shift) - 1 if shift > 0 else 0
            buffered_bits -= bits_per_symbol

    if len(values) != symbol_count:
        raise ValueError(f"Decoded {len(values)} symbols, expected {symbol_count}.")

    if buffered_bits != 0 or buffer != 0:
        raise ValueError("Found trailing non-zero bits in the raw bitstream payload.")

    return values


def _flatten_step_history(step_history: list[Any]) -> tuple[list[int], list[int]]:
    flat_symbols: list[int] = []
    step_lengths: list[int] = []

    for step in step_history:
        step_tensor = to_long_tensor(step).reshape(-1)
        step_lengths.append(int(step_tensor.numel()))
        flat_symbols.extend(int(value) for value in step_tensor.tolist())

    return flat_symbols, step_lengths


def _build_raw_bitstream_stats(
    image_id: str,
    file_path: Path,
    total_symbols: int,
    bits_per_symbol: int,
    image_size: int | tuple[int, int],
    num_timesteps: int,
) -> dict[str, Any]:
    image_area = _resolve_image_area(image_size)
    file_size_bytes = file_path.stat().st_size
    file_size_bits = file_size_bytes * 8
    return {
        "image_id": image_id,
        "num_timesteps": num_timesteps,
        "symbol_count": total_symbols,
        "bits_per_symbol": bits_per_symbol,
        "payload_bits": total_symbols * bits_per_symbol,
        "file_size_bytes": file_size_bytes,
        "file_size_bits": file_size_bits,
        "actual_bpp": file_size_bits / float(image_area),
    }


def split_index_history_by_batch(
    history: list[Any],
    num_timesteps: int,
    batch_size: int,
) -> list[list[torch.Tensor]]:
    if not history:
        return [[] for _ in range(batch_size)]

    tensors = [to_long_tensor(step) for step in history]
    per_batch = [[] for _ in range(batch_size)]

    if len(tensors) == num_timesteps:
        for tensor in tensors:
            if tensor.dim() == 0:
                if batch_size != 1:
                    raise ValueError("Scalar history entries require batch_size == 1.")
                tensor = tensor.reshape(1)
            elif batch_size == 1 and tensor.dim() >= 1 and tensor.shape[0] != 1:
                tensor = tensor.unsqueeze(0)

            if tensor.shape[0] != batch_size:
                raise ValueError(
                    f"Unexpected history tensor shape {tuple(tensor.shape)} for batch_size={batch_size}."
                )

            for batch_idx in range(batch_size):
                per_batch[batch_idx].append(tensor[batch_idx].clone())
        return per_batch

    if len(tensors) == num_timesteps * batch_size:
        for t in range(num_timesteps):
            step_slice = tensors[t * batch_size : (t + 1) * batch_size]
            for batch_idx, tensor in enumerate(step_slice):
                per_batch[batch_idx].append(tensor.clone())
        return per_batch

    raise ValueError(
        f"Unexpected history length {len(tensors)}; expected {num_timesteps} or {num_timesteps * batch_size}."
    )


def reconstruct_indices_per_batch(
    history: list[Any],
    num_timesteps: int,
    batch_size: int,
) -> tuple[torch.Tensor, bool]:
    if not history:
        raise ValueError("History is empty.")

    tensors = [to_long_tensor(step) for step in history]

    if len(tensors) == num_timesteps:
        first = tensors[0]

        if first.dim() == 3:
            if first.shape[0] != batch_size:
                raise ValueError(
                    f"Unexpected history tensor shape {tuple(first.shape)} for batch_size={batch_size}."
                )
            patch_max = max(step.size(-1) for step in tensors)
            steps = [
                step if step.size(-1) == patch_max else F.pad(step, (0, patch_max - step.size(-1)), value=-1)
                for step in tensors
            ]
            all_indices = torch.stack(steps, dim=0)
            return all_indices.permute(1, 0, 2, 3), True

        if first.dim() == 2:
            if batch_size == 1 and first.shape[0] != 1:
                patch_max = max(step.numel() for step in tensors)
                steps = [
                    F.pad(step.reshape(1, -1), (0, patch_max - step.numel()), value=-1) for step in tensors
                ]
                all_indices = torch.stack(steps, dim=0)
                return all_indices.permute(1, 0, 2), False

            if first.shape[0] != batch_size:
                raise ValueError(
                    f"Unexpected history tensor shape {tuple(first.shape)} for batch_size={batch_size}."
                )

            patch_max = max(step.size(-1) for step in tensors)
            steps = [
                step if step.size(-1) == patch_max else F.pad(step, (0, patch_max - step.size(-1)), value=-1)
                for step in tensors
            ]
            all_indices = torch.stack(steps, dim=0)
            return all_indices.permute(1, 0, 2), False

        if first.dim() == 1:
            if batch_size == 1:
                patch_max = max(step.numel() for step in tensors)
                steps = [
                    F.pad(step.reshape(1, -1), (0, patch_max - step.numel()), value=-1) for step in tensors
                ]
                all_indices = torch.stack(steps, dim=0)
                return all_indices.permute(1, 0, 2), False

            if first.shape[0] != batch_size:
                raise ValueError(
                    f"Unexpected history tensor shape {tuple(first.shape)} for batch_size={batch_size}."
                )

            steps = [step.reshape(batch_size, 1) for step in tensors]
            all_indices = torch.stack(steps, dim=0)
            return all_indices.permute(1, 0, 2), False

        raise ValueError(f"Unsupported history tensor rank: {first.dim()}")

    if len(tensors) == num_timesteps * batch_size:
        first = tensors[0]

        if first.dim() == 2:
            steps = []
            for t in range(num_timesteps):
                step_slice = tensors[t * batch_size : (t + 1) * batch_size]
                patch_max = max(step.size(1) for step in step_slice)
                padded = [
                    step if step.size(1) == patch_max else F.pad(step, (0, patch_max - step.size(1)), value=-1)
                    for step in step_slice
                ]
                steps.append(torch.stack(padded, dim=0))

            patch_max = max(step.size(2) for step in steps)
            steps = [
                step if step.size(2) == patch_max else F.pad(step, (0, patch_max - step.size(2)), value=-1)
                for step in steps
            ]
            all_indices = torch.stack(steps, dim=0)
            return all_indices.permute(1, 0, 2, 3), True

        if first.dim() == 1:
            steps = []
            for t in range(num_timesteps):
                step_slice = tensors[t * batch_size : (t + 1) * batch_size]
                patch_max = max(step.numel() for step in step_slice)
                padded = [
                    F.pad(step.reshape(-1), (0, patch_max - step.numel()), value=-1) for step in step_slice
                ]
                steps.append(torch.stack(padded, dim=0))

            patch_max = max(step.size(1) for step in steps)
            steps = [
                step if step.size(1) == patch_max else F.pad(step, (0, patch_max - step.size(1)), value=-1)
                for step in steps
            ]
            all_indices = torch.stack(steps, dim=0)
            return all_indices.permute(1, 0, 2), False

        raise ValueError(f"Unsupported history tensor rank: {first.dim()}")

    raise ValueError(
        f"Unexpected history length {len(tensors)}; expected {num_timesteps} or {num_timesteps * batch_size}."
    )


def build_history_rows(
    history: list[Any],
    num_timesteps: int,
    batch_size: int,
    batch_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    per_batch = split_index_history_by_batch(history, num_timesteps, batch_size)

    for batch_idx, batch_history in enumerate(per_batch):
        image_id = batch_ids[batch_idx] if batch_ids and batch_idx < len(batch_ids) else None
        for timestep, step in enumerate(batch_history):
            rows.append(
                {
                    "time_step": timestep,
                    "batch_index": batch_idx,
                    "image_id": image_id,
                    "tensor_shape": list(step.shape),
                    "indices": step.reshape(-1).tolist(),
                }
            )

    return rows


def save_scheduler_index_history(
    bin_dir: str | Path,
    batch_ids: list[str],
    history: list[Any],
    num_timesteps: int,
    batch_size: int,
    codebook_size: int,
    image_size: int | tuple[int, int],
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    del metadata

    bin_dir = Path(bin_dir)
    bin_dir.mkdir(parents=True, exist_ok=True)

    bits_per_symbol = get_fixed_width_bits(codebook_size)
    per_batch = split_index_history_by_batch(history, num_timesteps, batch_size)
    saved_stats: list[dict[str, Any]] = []

    for batch_idx, step_history in enumerate(per_batch):
        image_id = batch_ids[batch_idx] if batch_idx < len(batch_ids) else f"sample_{batch_idx}"
        flat_symbols, step_lengths = _flatten_step_history(step_history)
        total_bits = len(flat_symbols) * bits_per_symbol
        if total_bits % 8 != 0:
            raise ValueError(
                f"Bitstream for {image_id} is not byte-aligned: total_bits={total_bits}, "
                f"step_lengths={step_lengths}, bits_per_symbol={bits_per_symbol}."
            )

        payload = _pack_symbols(flat_symbols, bits_per_symbol)
        file_path = bin_dir / f"{image_id}.bin"
        file_path.write_bytes(payload)
        saved_stats.append(
            _build_raw_bitstream_stats(
                image_id=image_id,
                file_path=file_path,
                total_symbols=len(flat_symbols),
                bits_per_symbol=bits_per_symbol,
                image_size=image_size,
                num_timesteps=len(step_history),
            )
        )

    return saved_stats


def load_batch_histories(
    bin_dir: str | Path,
    batch_ids: list[str],
    scheduler: Any,
    timesteps: Iterable[Any],
    codebook_size: int,
) -> list[list[torch.Tensor]]:
    histories: list[list[torch.Tensor]] = []
    bin_dir = Path(bin_dir)

    summary = summarize_expected_bitstream(
        scheduler=scheduler,
        timesteps=timesteps,
        codebook_size=codebook_size,
        image_size=(1, 1),
    )
    step_symbol_counts = list(summary["step_symbol_counts"])
    total_symbols = int(summary["total_symbols"])
    expected_bytes = int(summary["total_bytes"])
    bits_per_symbol = int(summary["bits_per_symbol"])

    for image_id in batch_ids:
        bin_path = bin_dir / f"{image_id}.bin"
        if not bin_path.exists():
            raise FileNotFoundError(f"Binary file not found: {bin_path}")

        payload = bin_path.read_bytes()
        if len(payload) == expected_bytes:
            flat_symbols = _unpack_symbols(payload, bits_per_symbol=bits_per_symbol, symbol_count=total_symbols)
            cursor = 0
            step_history: list[torch.Tensor] = []
            for step_symbols in step_symbol_counts:
                step_values = flat_symbols[cursor : cursor + step_symbols]
                step_history.append(torch.tensor(step_values, dtype=torch.long))
                cursor += step_symbols

            if cursor != len(flat_symbols):
                raise ValueError(f"Decoded cursor mismatch for {bin_path}: {cursor} vs {len(flat_symbols)}.")

            histories.append(step_history)
            continue

        with open(bin_path, "rb") as handle:
            payload_obj = pickle.load(handle)

        if isinstance(payload_obj, dict) and "data" in payload_obj:
            history_data = payload_obj["data"]
        else:
            history_data = payload_obj

        histories.append([deserialize_history_step(step) for step in history_data])

    return histories


def stack_step_history(step_entries: list[Any], device: torch.device | str | None = None) -> torch.Tensor:
    tensors = [deserialize_history_step(step) for step in step_entries]
    stacked = torch.stack(tensors, dim=0).long()
    return stacked.to(device=device) if device is not None else stacked
