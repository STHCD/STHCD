import numpy as np
import torch
from typing import Dict, Optional, Union

from skimage.metrics import peak_signal_noise_ratio
from torchmetrics.image.dists import DeepImageStructureAndTextureSimilarity
from utils.schedule_registry import resolve_schedule_payload

LATENT_CHANNELS = 4.0


def _resolve_image_area(H: int, W: int) -> float:
    if H <= 0 or W <= 0:
        raise ValueError("H and W must be positive.")
    return float(H * W)


def _normalize_codebook_channel(codebook_channel: int) -> int:
    if codebook_channel <= 0:
        raise ValueError("codebook_channel must be positive.")
    # Preserve the historical baseline behavior used by the project.
    return 2 if codebook_channel == 3 else codebook_channel


def _parse_ratio_schedule(
    schedule_ref: Optional[Union[str, Dict[str, object]]],
    field_name: str,
    T: int,
) -> Dict[int, int]:
    if not schedule_ref:
        return {}

    schedule_payload = resolve_schedule_payload(schedule_ref)
    raw_map = schedule_payload.get(field_name, {})
    return {int(float(k) * T): int(v) for k, v in raw_map.items()}


def _resolve_step_value(step_index: int, schedule: Dict[int, int], default_value: int) -> int:
    for threshold, value in schedule.items():
        if step_index >= threshold:
            return int(value)
    return int(list(schedule.values())[-1]) if schedule else int(default_value)


def calculate_bpp(
    T: int = 10,
    K: Optional[int] = 256,
    H: int = 256,
    W: int = 256,
    residual_unshuffle_factor: int = 2,
    codebook_channel: int = 4,
    schedule_payload: Optional[Union[str, Dict[str, object]]] = None,
) -> float:
    if T <= 0:
        return 0.0
    if K is None:
        raise ValueError("K must be provided.")
    if residual_unshuffle_factor <= 0:
        raise ValueError("residual_unshuffle_factor must be positive.")
    codebook_channel = _normalize_codebook_channel(codebook_channel)

    image_area = _resolve_image_area(H=H, W=W)
    bits_per_symbol = float(np.log2(K))
    shuffle_schedule = _parse_ratio_schedule(schedule_payload, "shuffle_factor_map", T)
    channel_schedule = _parse_ratio_schedule(schedule_payload, "codebook_channel_map", T)

    total_bits = 0.0
    for step_index in range(T - 1, -1, -1):
        current_shuffle_factor = _resolve_step_value(step_index, shuffle_schedule, residual_unshuffle_factor)
        current_channel = _resolve_step_value(step_index, channel_schedule, codebook_channel)
        current_channel = min(current_channel, codebook_channel)

        step_symbols = LATENT_CHANNELS * float(current_shuffle_factor ** 2) / float(current_channel)
        total_bits += bits_per_symbol * step_symbols

    return total_bits / image_area


def calculate_lpips(image1_path, image2, lpips_fn, device):
    try:
        from PIL import Image
        import torchvision.transforms as transforms

        if isinstance(image1_path, str):
            img1 = Image.open(image1_path).convert("RGB")
        else:
            img1 = image1_path.convert("RGB") if hasattr(image1_path, "convert") else image1_path

        if isinstance(image2, str):
            img2 = Image.open(image2).convert("RGB")
        else:
            img2 = image2.convert("RGB") if hasattr(image2, "convert") else image2

        if img1.size != img2.size:
            img2 = img2.resize(img1.size, Image.Resampling.LANCZOS)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        img1_tensor = transform(img1).unsqueeze(0).to(device)
        img2_tensor = transform(img2).unsqueeze(0).to(device)

        with torch.no_grad():
            lpips_score = lpips_fn(img1_tensor, img2_tensor).item()

        return lpips_score

    except Exception as e:
        print(f"Error calculating LPIPS: {e}")
        return 1.0


def calculate_dists(image1_path, image2, device):
    try:
        from PIL import Image
        import torchvision.transforms as transforms

        if isinstance(image1_path, str):
            img1 = Image.open(image1_path).convert("RGB")
        else:
            img1 = image1_path.convert("RGB") if hasattr(image1_path, "convert") else image1_path

        if isinstance(image2, str):
            img2 = Image.open(image2).convert("RGB")
        else:
            img2 = image2.convert("RGB") if hasattr(image2, "convert") else image2

        if img1.size != img2.size:
            img2 = img2.resize(img1.size, Image.Resampling.LANCZOS)

        transform = transforms.ToTensor()

        img1_tensor = transform(img1).unsqueeze(0).to(device).float()
        img2_tensor = transform(img2).unsqueeze(0).to(device).float()

        dists_fn = DeepImageStructureAndTextureSimilarity().to(device)
        with torch.no_grad():
            dists_score = dists_fn(img1_tensor, img2_tensor).item()

        return dists_score

    except Exception as e:
        print(f"Error calculating DISTS: {e}")
        return 1.0


def calculate_psnr(image1_path, image2_path, data_range=255.0):
    try:
        from PIL import Image

        if isinstance(image1_path, str):
            img1 = np.array(Image.open(image1_path).convert("RGB"))
        else:
            img1 = np.array(image1_path.convert("RGB") if hasattr(image1_path, "convert") else image1_path)

        if isinstance(image2_path, str):
            img2 = np.array(Image.open(image2_path).convert("RGB"))
        else:
            img2 = np.array(image2_path.convert("RGB") if hasattr(image2_path, "convert") else image2_path)

        if img1.shape != img2.shape:
            from PIL import Image as PILImage

            img2_pil = PILImage.fromarray(img2)
            img2_pil = img2_pil.resize((img1.shape[1], img1.shape[0]), PILImage.Resampling.LANCZOS)
            img2 = np.array(img2_pil)

        return peak_signal_noise_ratio(img1, img2, data_range=data_range)

    except Exception as e:
        print(f"Error calculating PSNR: {e}")
        return 0.0
