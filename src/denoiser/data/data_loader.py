from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy.typing as npt

    from denoiser.configs.config import PairingKeyWords

from denoiser.utils.data_utils import hwc_to_chw


class PairdDataset(Dataset):
    """Load dataset.

    Args:
        data_paths: Root directory of images.
        data_loading_fn: Function to load image from path.
        paring_fn: Function to get paired image paths of clean and noisy.
        img_standardization_fn: Function to standardize image.
        data_augmentation_fn: Function to augment image.
        noise_sigma: If >0, add Gaussian noise with given sigma as std.
        limit: if given, limit number of images to load.

    """

    def __init__(
        self,
        data_paths: Path | list[Path],
        data_loading_fn: Callable[[Path], npt.NDArray[np.uint8]],
        img_standardization_fn: Callable[[npt.NDArray[np.uint8]], npt.NDArray[np.float32]],
        paring_fn: Callable[[Path], npt.NDArray[np.uint8]] | None = None,
        data_augmentation_fn: Callable[[npt.NDArray[np.uint8]], npt.NDArray[np.uint8]] | None = None,
        img_read_keywards: PairingKeyWords | None = None,
        noise_sigma: float = 0.08,
        limit: int | None = None,
    ) -> None:
        self.data_loading_fn = data_loading_fn
        self.img_standardization_fn = img_standardization_fn
        self.paring_fn = paring_fn if paring_fn is not None else add_gaussian_noise_fn(noise_sigma)
        self.data_augmentation_fn = data_augmentation_fn
        self.img_read_keywards = img_read_keywards
        self.noise_sigma = noise_sigma

        exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif"]
        if isinstance(data_paths, Path):
            data_paths = [data_paths]

        self.img_paths: list[Path] = [
            p
            for data_dir in data_paths
            for p in data_dir.rglob("*")
            if p.suffix.lower() in exts and _match_keywords(p, self.img_read_keywards)
        ][:limit]

        self.img_paths = sorted(self.img_paths)
        self.mean, self.std = None, None

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Get (clean, noisy) pair.

        Pair the keywords noisy and clean appropriately in your paring_fn.
        """
        clean_path = self.img_paths[idx % len(self.img_paths)]
        img_clean = self.data_loading_fn(clean_path)
        img_noisy = self.paring_fn(clean_path)
        if self.data_augmentation_fn is not None:
            img_clean = self.data_augmentation_fn(img_clean)
            img_noisy = self.data_augmentation_fn(img_noisy)  # it is different seed?

        img_clean = self.img_standardization_fn(img_clean)
        img_noisy = self.img_standardization_fn(img_noisy)  # it is diffrent params?

        img_clean = hwc_to_chw(img_clean)
        img_noisy = hwc_to_chw(img_noisy)

        return img_clean, img_noisy


def add_gaussian_noise_fn(sigma: float) -> Callable[[npt.NDArray[np.float32]], npt.NDArray[np.float32]]:
    """Return function to add Gaussian noise with given sigma."""

    def _add_gaussian_noise(img: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        rng = np.random.default_rng()
        noise = rng.standard_normal(img.shape) * sigma / 255.0
        noisy_img = img + noise
        noisy_img = torch.clamp(noisy_img, 0.0, 1.0)
        return noisy_img

    return _add_gaussian_noise


def _match_keywords(p: Path, keywords: PairingKeyWords | None) -> bool:
    if keywords is None:
        return True

    if keywords.detector is not None:
        return keywords.clean in p.stem and keywords.detector[0] in p.stem
    return keywords.clean in p.stem
