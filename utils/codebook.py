import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from utils.runtime_paths import build_codebook_cache_dir, get_repo_root


def _display_cache_path(path: Path) -> str:
    try:
        return str(path.relative_to(get_repo_root()))
    except ValueError:
        return str(path)


class _DynamicCodebookWithCache(nn.Module):
    """Shared cached codebook implementation for timestep-indexed random codebooks."""

    distribution_name = "random"

    def __init__(
        self,
        num_timesteps,
        codebook_size,
        image_shape,
        cache_dir,
        batch_size,
        seed=42,
        data_type="float16",
        load_on_init=True,
    ):
        super().__init__()
        if data_type not in (None, "float16"):
            raise ValueError("STHCD only supports float16 codebook caching.")
        self.num_timesteps = num_timesteps
        self.codebook_size = codebook_size
        self.image_shape = image_shape
        self.seed = seed
        self.data_type = "float16"
        self.dtype = torch.float16
        self.np_dtype = np.float16
        self.cache_dir = Path(cache_dir)
        self.batch_size = batch_size
        self.elements_per_codebook = self.codebook_size * np.prod(self.image_shape).item()

        os.makedirs(self.cache_dir, exist_ok=True)

        self.cache_files = []
        for batch_idx in range((num_timesteps + 1 + batch_size - 1) // batch_size):
            start_t = batch_idx * batch_size
            end_t = min(start_t + batch_size, num_timesteps + 1)
            self.cache_files.append(f"seed{self.seed}_codebook_t{start_t}_to_t{end_t - 1}.npy")

        all_files_exist = all((self.cache_dir / cache_file).exists() for cache_file in self.cache_files)
        if load_on_init and not all_files_exist:
            print(
                f"Missing {self.distribution_name} codebook cache in {_display_cache_path(self.cache_dir)}, generating it..."
            )
            self.generate_all_codebooks()
        elif load_on_init and all_files_exist:
            print(f"Reusing {self.distribution_name} codebook cache from {_display_cache_path(self.cache_dir)}.")

        self.current_cache = None
        self.current_batch_idx = -1

    def _sample(self, numel: int, generator: torch.Generator) -> torch.Tensor:
        raise NotImplementedError

    def _get_batch_idx(self, timestep):
        """Return the cache batch index for a given timestep."""
        return timestep // self.batch_size

    def generate_all_codebooks(self):
        """Generate and cache every timestep codebook."""
        print(f"Generating all {self.distribution_name} codebooks into {_display_cache_path(self.cache_dir)}...")

        generator = torch.Generator().manual_seed(self.seed)
        for batch_idx, cache_file in enumerate(self.cache_files):
            start_t = batch_idx * self.batch_size
            end_t = min(start_t + self.batch_size, self.num_timesteps + 1)
            batch_codebooks = np.zeros(
                (end_t - start_t, self.codebook_size, *self.image_shape),
                dtype=self.np_dtype,
            )

            for i, _ in enumerate(tqdm(range(start_t, end_t), desc=f"Generating codebooks {start_t}-{end_t - 1}")):
                codebook = self._sample(self.elements_per_codebook, generator).reshape(
                    self.codebook_size, *self.image_shape
                )
                batch_codebooks[i] = codebook.cpu().numpy().astype(self.np_dtype, copy=False)

            cache_path = self.cache_dir / cache_file
            np.save(cache_path, batch_codebooks)
            print(f"Saved timesteps {start_t} to {end_t - 1} into {cache_file}")

    def _load_batch(self, batch_idx):
        """Load one cached timestep batch into memory."""
        if batch_idx == self.current_batch_idx and self.current_cache is not None:
            return

        cache_path = self.cache_dir / self.cache_files[batch_idx]
        if not cache_path.exists():
            self.generate_batch(batch_idx)

        self.current_cache = np.load(cache_path)
        self.current_batch_idx = batch_idx

    def generate_batch(self, batch_idx):
        """Generate and save a single cache batch."""
        start_t = batch_idx * self.batch_size
        end_t = min(start_t + self.batch_size, self.num_timesteps + 1)
        cache_file = self.cache_files[batch_idx]

        generator = torch.Generator().manual_seed(self.seed)
        for _ in range(start_t):
            _ = self._sample(self.elements_per_codebook, generator)

        batch_codebooks = np.zeros(
            (end_t - start_t, self.codebook_size, *self.image_shape),
            dtype=self.np_dtype,
        )

        for i, _ in enumerate(range(start_t, end_t)):
            codebook = self._sample(self.elements_per_codebook, generator).reshape(
                self.codebook_size, *self.image_shape
            )
            batch_codebooks[i] = codebook.cpu().numpy().astype(self.np_dtype, copy=False)

        cache_path = self.cache_dir / cache_file
        np.save(cache_path, batch_codebooks)
        print(f"Generated and saved timesteps {start_t} to {end_t - 1} into {cache_file}")

    def __getitem__(self, timestep):
        """Return the codebook for a single timestep."""
        if timestep < 0 or timestep > self.num_timesteps:
            raise IndexError(f"Timestep index {timestep} is out of range [0, {self.num_timesteps}].")

        batch_idx = self._get_batch_idx(timestep)
        self._load_batch(batch_idx)
        relative_idx = timestep - batch_idx * self.batch_size
        return torch.from_numpy(self.current_cache[relative_idx]).to(self.dtype)


class DynamicGaussianCodebookWithCache(_DynamicCodebookWithCache):
    """Gaussian codebook cache used by the standard STHCD scheduler family."""

    distribution_name = "Gaussian"

    def _sample(self, numel: int, generator: torch.Generator) -> torch.Tensor:
        return torch.randn(numel, generator=generator, dtype=self.dtype)


class DynamicLaplaceCodebookWithCache(_DynamicCodebookWithCache):
    """Laplace codebook cache used by experimental hierarchical schedulers."""

    distribution_name = "Laplace"

    def _sample(self, numel: int, generator: torch.Generator) -> torch.Tensor:
        uniform = torch.rand(numel, generator=generator, dtype=torch.float32)
        laplace = torch.where(
            uniform < 0.5,
            torch.log(torch.clamp(uniform * 2.0, min=1e-12)),
            -torch.log(torch.clamp((1.0 - uniform) * 2.0, min=1e-12)),
        )
        return laplace.to(self.dtype)


if __name__ == "__main__":
    codebook_size = 256
    codebook_entries = 1001
    codebook_dims = (4, 64, 64)
    data_type = "float16"

    codebook = DynamicGaussianCodebookWithCache(
        num_timesteps=codebook_entries - 1,
        codebook_size=codebook_size,
        image_shape=codebook_dims,
        batch_size=codebook_entries,
        cache_dir=build_codebook_cache_dir(
            data_type=data_type,
            variant="codebook",
            cache_name=f"codebook_cache_{codebook_size}_{codebook_entries - 1}_{codebook_dims[-1]}",
        ),
        data_type=data_type,
        load_on_init=True,
    )
