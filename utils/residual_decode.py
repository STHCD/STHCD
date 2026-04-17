from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F

from utils.residual_transform import residual_restore


def adapt_codebook_spatial(
    codebook_t: torch.Tensor,
    target_height: int,
    target_width: int,
    allow_resize: bool,
) -> torch.Tensor:
    _, _, codebook_height, codebook_width = codebook_t.shape
    if codebook_height == target_height and codebook_width == target_width:
        return codebook_t

    if not allow_resize:
        raise ValueError(
            f"Spatial mismatch after residual unshuffle: {(target_height, target_width)} vs codebook {(codebook_height, codebook_width)}"
        )

    if target_height <= codebook_height and target_width <= codebook_width:
        return codebook_t[:, :, :target_height, :target_width]

    return F.interpolate(
        codebook_t,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )


def build_sliding_windows(
    total_channels: int,
    codebook_channels: int,
    device: torch.device,
    stride: int = 1,
    prefer_ch0_when_sliding: bool = True,
) -> list[torch.Tensor]:
    if total_channels < codebook_channels:
        raise ValueError(
            f"Channel mismatch: total_channels={total_channels} < codebook_channels={codebook_channels}."
        )

    stride = max(int(stride), 1)
    last_start = total_channels - codebook_channels
    windows = [
        torch.arange(start, start + codebook_channels, device=device, dtype=torch.long)
        for start in range(0, last_start + 1, stride)
    ]

    if prefer_ch0_when_sliding and windows:
        window = windows[-1].tolist()
        window[0] = 0

        dedup: list[int] = []
        seen = set()
        for channel in window:
            if channel not in seen:
                dedup.append(channel)
                seen.add(channel)

        if len(dedup) < codebook_channels:
            for channel in range(total_channels - 1, -1, -1):
                if channel not in seen:
                    dedup.append(channel)
                    seen.add(channel)
                    if len(dedup) == codebook_channels:
                        break

        windows[-1] = torch.tensor(dedup[:codebook_channels], device=device, dtype=torch.long)

    return windows


def reconstruct_quantized_noise(
    idx_codebook: torch.Tensor,
    codebook_t: torch.Tensor,
    sample_shape: torch.Size,
    residual_factor: int,
    allow_spatial_resize: bool = False,
    slide_stride: int = 1,
    prefer_ch0_when_sliding: bool = True,
) -> torch.Tensor:
    if idx_codebook.dim() == 1:
        idx_codebook = idx_codebook.unsqueeze(1)
    if idx_codebook.dim() != 2:
        raise ValueError(f"Expected idx_codebook to have rank 1 or 2, got shape {tuple(idx_codebook.shape)}")

    batch_size, latent_channels, latent_height, latent_width = sample_shape
    target_channels = latent_channels * (residual_factor**2)
    target_height = latent_height // residual_factor
    target_width = latent_width // residual_factor

    codebook_t = adapt_codebook_spatial(
        codebook_t,
        target_height=target_height,
        target_width=target_width,
        allow_resize=allow_spatial_resize,
    )

    _, codebook_channels, _, _ = codebook_t.shape
    idx_codebook = idx_codebook.to(device=codebook_t.device, dtype=torch.long)

    if idx_codebook.shape[0] != batch_size:
        raise ValueError(
            f"Batch mismatch between indices {tuple(idx_codebook.shape)} and sample shape {tuple(sample_shape)}."
        )

    if target_channels % codebook_channels == 0 and idx_codebook.shape[1] == target_channels // codebook_channels:
        quantized_slices = [codebook_t[idx_codebook[:, slice_idx]] for slice_idx in range(idx_codebook.shape[1])]
        z_tilde = torch.cat(quantized_slices, dim=1)
    else:
        windows = build_sliding_windows(
            total_channels=target_channels,
            codebook_channels=codebook_channels,
            device=codebook_t.device,
            stride=slide_stride,
            prefer_ch0_when_sliding=prefer_ch0_when_sliding,
        )
        if idx_codebook.shape[1] != len(windows):
            raise ValueError(
                f"Index/window mismatch: got {idx_codebook.shape[1]} windows, expected {len(windows)}."
            )

        z_accum = torch.zeros(
            (batch_size, target_channels, target_height, target_width),
            device=codebook_t.device,
            dtype=codebook_t.dtype,
        )
        counts = torch.zeros(
            (1, target_channels, 1, 1),
            device=codebook_t.device,
            dtype=codebook_t.dtype,
        )

        for slice_idx, channel_window in enumerate(windows):
            quantized = codebook_t[idx_codebook[:, slice_idx]]
            z_accum.index_add_(1, channel_window, quantized)
            counts.index_add_(
                1,
                channel_window,
                torch.ones(
                    (1, channel_window.numel(), 1, 1),
                    device=codebook_t.device,
                    dtype=codebook_t.dtype,
                ),
            )

        z_tilde = z_accum / counts.clamp_min(1.0)

    return residual_restore(z_tilde, latent_height, latent_width)
