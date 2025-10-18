from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from collections.abc import Callable

    from denoiser.configs.config import PairingKeyWords

from denoiser.utils.data_utils import hwc_to_chw


class PairedDataset(Dataset[tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]]):
    """Load dataset.

    Args:
        data_paths: Root directory of images.
        data_loading_fn: Function to load image from path.
        pairing_fn: Function to get paired image paths of clean and noisy.
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
        pairing_fn: Callable[[Path], npt.NDArray[np.uint8]],
        data_augmentation_fn: Callable[
            [npt.NDArray[np.uint8], npt.NDArray[np.uint8]], tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]
        ]
        | None = None,
        img_read_keywords: PairingKeyWords | None = None,
        noise_sigma: float = 0.08,
        limit: int | None = None,
    ) -> None:
        self.data_loading_fn = data_loading_fn
        self.img_standardization_fn = img_standardization_fn
        self.pairing_fn = pairing_fn
        self.data_augmentation_fn = data_augmentation_fn
        self.img_read_keywords = img_read_keywords
        self.noise_sigma = noise_sigma

        exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif"]
        if isinstance(data_paths, Path):
            data_paths = [data_paths]

        self.img_paths: list[Path] = [
            p
            for data_dir in data_paths
            for p in data_dir.rglob("*")
            if p.suffix.lower() in exts and match_keywords(p, self.img_read_keywords)
        ][:limit]

        self.img_paths = sorted(self.img_paths)
        self.mean, self.std = None, None

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
        clean_path = self.img_paths[idx % len(self.img_paths)]
        img_clean = self.data_loading_fn(clean_path)
        img_noisy = self.pairing_fn(clean_path)
        return img_clean, img_noisy


class TrainSubset(
    Dataset[tuple[npt.NDArray[np.uint8] | npt.NDArray[np.float32], npt.NDArray[np.uint8] | npt.NDArray[np.float32]]]
):
    """Subset of a dataset for training."""

    def __init__(self, dataset: PairedDataset, indices: list[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(
        self, idx: int
    ) -> tuple[npt.NDArray[np.uint8] | npt.NDArray[np.float32], npt.NDArray[np.uint8] | npt.NDArray[np.float32]]:
        img_clean, img_noisy = self.dataset[self.indices[idx]]

        if self.dataset.data_augmentation_fn is not None:
            img_clean, img_noisy = self.dataset.data_augmentation_fn(img_clean, img_noisy)

        img_clean = self.dataset.img_standardization_fn(img_clean)
        img_noisy = self.dataset.img_standardization_fn(img_noisy)

        img_clean = hwc_to_chw(img_clean)
        img_noisy = hwc_to_chw(img_noisy)

        return img_clean, img_noisy


class ValSubset(
    Dataset[tuple[npt.NDArray[np.uint8] | npt.NDArray[np.float32], npt.NDArray[np.uint8] | npt.NDArray[np.float32]]]
):
    """Subset of a dataset for validation."""

    def __init__(self, dataset: PairedDataset, indices: list[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(
        self, idx: int
    ) -> tuple[npt.NDArray[np.uint8] | npt.NDArray[np.float32], npt.NDArray[np.uint8] | npt.NDArray[np.float32]]:
        img_clean, img_noisy = self.dataset[self.indices[idx]]

        img_clean = self.dataset.img_standardization_fn(img_clean)
        img_noisy = self.dataset.img_standardization_fn(img_noisy)

        img_clean = hwc_to_chw(img_clean)
        img_noisy = hwc_to_chw(img_noisy)

        return img_clean, img_noisy


def match_keywords(p: Path, keywords: PairingKeyWords | None) -> bool:
    if keywords is None:
        return True

    if keywords.detector is not None:
        return keywords.clean in p.stem and keywords.detector[0] in p.stem
    return keywords.clean in p.stem
