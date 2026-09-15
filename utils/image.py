import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import glob
import imageio.v2 as imageio

def read_png_uint8(png_path: str):
    png = imageio.imread(png_path)
    if png.ndim == 2:
        png = np.stack([png, png, png], axis=-1)
    png = png[..., :3]
    if png.dtype != np.uint8:
        mx = float(np.max(png)) if np.max(png) > 0 else 1.0
        png = (png.astype(np.float64) / mx * 255.0 + 0.5).astype(np.uint8)
    return png

@torch.no_grad()
def cdf_match_to_png_uint8(src_chw_float: torch.Tensor,
                           tgt_hwc_uint8: np.ndarray,
                           n_quant: int = 2048,
                           clip_lo: float = 0.0005,
                           clip_hi: float = 0.9995,
                           eps: float = 1e-12) -> torch.Tensor:
    """
    src_chw_float: torch.Tensor [3,H,W], float
    tgt_hwc_uint8: numpy uint8 [H,W,3]
    return: torch.FloatTensor [3,H,W] in [0,1] (mapped via target PNG distribution)
    """
    assert src_chw_float.ndim == 3 and src_chw_float.shape[0] == 3
    H, W = src_chw_float.shape[1], src_chw_float.shape[2]
    assert tgt_hwc_uint8.shape[0] == H and tgt_hwc_uint8.shape[1] == W

    device = src_chw_float.device
    out = torch.empty_like(src_chw_float, dtype=torch.float32)

    q = torch.linspace(clip_lo, clip_hi, n_quant, device=device)

    for c in range(3):
        src = src_chw_float[c].flatten().float()
        tgt = torch.from_numpy(tgt_hwc_uint8[..., c]).to(device).flatten().float()

        # Remove non-finite values before estimating the CDF.
        src = src[torch.isfinite(src)]
        tgt = tgt[torch.isfinite(tgt)]
        if src.numel() < 10 or tgt.numel() < 10:
            out[c].zero_()
            continue

        src_q = torch.quantile(src, q)
        tgt_q = torch.quantile(tgt, q)

        # Deduplicate flat quantile segments to keep interpolation stable.
        keep = torch.ones_like(src_q, dtype=torch.bool)
        keep[1:] = (src_q[1:] - src_q[:-1]).abs() > eps
        src_q2 = src_q[keep]
        tgt_q2 = tgt_q[keep]

        if src_q2.numel() < 2:
            out[c].fill_(float(torch.median(tgt) / 255.0))
            continue

        # Torch does not expose a direct interpolation utility for this case, so use NumPy once per image.
        src_np = src_chw_float[c].detach().cpu().numpy()
        x = src_np.reshape(-1)
        xp = src_q2.detach().cpu().numpy()
        fp = tgt_q2.detach().cpu().numpy()

        mapped = np.interp(x, xp, fp)
        mapped = np.clip(mapped, 0.0, 255.0).reshape(H, W)
        out[c] = torch.from_numpy(mapped).to(device).float() / 255.0

    return out

class ImageDataset(Dataset):
    def __init__(self, image_dir, image_size=256):
        self.image_dir = image_dir
        self.image_size = image_size

        # Get all image files
        self.image_paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            self.image_paths.extend(glob.glob(os.path.join(image_dir, ext)))

        self.image_ids = [os.path.splitext(os.path.basename(path))[0] for path in self.image_paths]

        print(f"Loaded {len(self.image_paths)} samples from {image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_id = self.image_ids[idx]

        # Load main image and ensure it's a proper tensor
        input_tensor = read_image(image_path)

        return {
            'image': input_tensor,
            'image_id': image_id,
            'image_path': image_path
        }

class HSIImageDataset(Dataset):
    def __init__(self, image_dir, image_size=256):
        self.image_dir = image_dir
        self.image_size = image_size

        # Keep scanning image previews while preferring lossless `.npy` payloads when available.
        self.image_paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            self.image_paths.extend(glob.glob(os.path.join(image_dir, ext)))

        # Keep the ordering deterministic because grouped samples must stay aligned.
        self.image_paths.sort()

        self.image_ids = [os.path.splitext(os.path.basename(path))[0] for path in self.image_paths]

        print(f"Loaded {len(self.image_paths)} samples from {image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_id = self.image_ids[idx]

        # Prefer the lossless `.npy` tensor if it exists next to the preview image.
        npy_path = os.path.splitext(image_path)[0] + '.npy'

        if os.path.exists(npy_path):
            numpy_img = np.load(npy_path).astype(np.float32)

            # Convert from HWC NumPy layout to CHW PyTorch layout.
            input_tensor = torch.from_numpy(numpy_img).permute(2, 0, 1)
        else:
            # read_image already maps preview pixels into [0, 1].
            input_tensor = read_image(image_path)

        return {
            'image': input_tensor,
            'image_id': image_id,
            'image_path': image_path  # Keep the preview path for bookkeeping.
        }

class SARImageDataset(Dataset):
    def __init__(self, image_dir, image_size=256):
        self.image_dir = image_dir
        self.image_size = image_size

        # Keep scanning image previews while preferring lossless `.npy` payloads when available.
        self.image_paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            self.image_paths.extend(glob.glob(os.path.join(image_dir, ext)))

        # Keep the ordering deterministic because grouped samples must stay aligned.
        self.image_paths.sort()

        self.image_ids = [os.path.splitext(os.path.basename(path))[0] for path in self.image_paths]

        print(f"Loaded {len(self.image_paths)} samples from {image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_id = self.image_ids[idx]

        # Prefer the lossless `.npy` tensor if it exists next to the preview image.
        npy_path = os.path.splitext(image_path)[0] + '.npy'

        if os.path.exists(npy_path):
            numpy_img = np.load(npy_path).astype(np.float32)

            input_tensor = torch.from_numpy(numpy_img).permute(2, 0, 1)

            vmin = float(input_tensor.min())
            vmax = float(input_tensor.max())
            eps = 1e-6
            already_01 = (vmin >= -eps) and (vmax <= 1.0 + eps)

            if not already_01:
                # Match the dynamic range to the paired PNG preview and map it back into [0, 1].
                tgt_png = read_png_uint8(image_path)
                input_tensor = cdf_match_to_png_uint8(input_tensor, tgt_png, n_quant=2048)
        else:
            # read_image already maps preview pixels into [0, 1].
            input_tensor = read_image(image_path)

        return {
            'image': input_tensor,
            'image_id': image_id,
            'image_path': image_path  # Keep the preview path for bookkeeping.
        }

class BinaryFileDataset(Dataset):
    def __init__(self, bin_dir, original_image_dir, image_size=256):
        self.bin_dir = bin_dir
        self.original_image_dir = original_image_dir
        self.image_size = image_size

        # Get all .bin files
        self.bin_paths = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))

        # Extract image IDs from .bin filenames
        self.image_ids = [os.path.splitext(os.path.basename(path))[0] for path in self.bin_paths]

        # Find corresponding original images
        self.original_image_paths = []
        for image_id in self.image_ids:
            original_path = None
            for ext in ['png', 'jpg', 'jpeg']:
                candidate_path = os.path.join(original_image_dir, f"{image_id}.{ext}")
                if os.path.exists(candidate_path):
                    original_path = candidate_path
                    break
            
            if original_path is None:
                print(f"Warning: missing reference image for {image_id}")
                self.original_image_paths.append(None)
            else:
                self.original_image_paths.append(original_path)
        
        if self.original_image_dir is not None:
            valid_indices = [i for i, path in enumerate(self.original_image_paths) if path is not None]
            self.bin_paths = [self.bin_paths[i] for i in valid_indices]
            self.image_ids = [self.image_ids[i] for i in valid_indices]
            self.original_image_paths = [self.original_image_paths[i] for i in valid_indices]

    def __len__(self):
        return len(self.bin_paths)

    def __getitem__(self, idx):
        bin_path = self.bin_paths[idx]
        image_id = os.path.splitext(os.path.basename(bin_path))[0]

        # Resolve the original image path if available.
        original_image_path = None
        if self.original_image_dir:
            # Try different image extensions
            for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                potential_path = os.path.join(self.original_image_dir, f"{image_id}{ext}")
                if os.path.exists(potential_path):
                    original_image_path = potential_path
                    break

        return {
            'bin_path': bin_path,
            'image_id': image_id,
            'original_image_path': original_image_path
        }

def read_image(image_path):
    input_image = Image.open(image_path)
    image_np = np.array(input_image).astype(np.float32) / 255.0
    # image_np = image_np * 2 - 1  # Scale from [0, 1] to [-1, 1]
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)  # Change from [H, W, C] to [C, H, W]
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension -> [1, C, H, W]
    return image_tensor
    
def save_image(decoded_image, output_dir, output_filename):
    """
    Save the decoded image to the specified directory.

    Args:
        decoded_image (torch.Tensor): The decoded image tensor.
        output_dir (str): The directory where the image will be saved.
        output_filename (str): The filename for the decoded image.
    """
    # Convert to numpy array if it's a torch tensor
    if isinstance(decoded_image, torch.Tensor):
        decoded_image = decoded_image.cpu().numpy()

    # Convert from [-1, 1] to [0, 1]
    if decoded_image.min() < 0:
        decoded_image = (decoded_image + 1) / 2  # Convert from [-1, 1] to [0, 1]

    # Rearrange to (H, W, C) format
    if decoded_image.shape[0] == 1 and len(decoded_image.shape) == 4:
        decoded_image = decoded_image.squeeze(0)

    if decoded_image.shape[0] == 3:
        decoded_image_np = 255. * np.transpose(decoded_image, (1, 2, 0))  # Change from [C, H, W] to [H, W, C]
    else:
        decoded_image_np = 255. * decoded_image

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save to the specified output directory
    output_path = os.path.join(output_dir, f"{output_filename}.png")
    img = Image.fromarray(decoded_image_np.astype(np.uint8))
    img.save(output_path)
    return output_path

def resize_image(image_path):
    image = Image.open(image_path)
    width, height = image.size
    new_width = 256
    new_height = int((new_width / width) * height)
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    return resized_image
