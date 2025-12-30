from __future__ import annotations

import json
import random
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Literal


from denoiser.utils.data_utils import hwc_to_chw


class PairedDataset(
    Dataset[tuple[npt.NDArray[np.uint8] | npt.NDArray[np.float32], npt.NDArray[np.uint8] | npt.NDArray[np.float32]]]
):
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
        mode: Literal["train", "val"] = "train",
        noise_sigma: float = 0.08,
        limit: int | None = None,
    ) -> None:
        self.data_loading_fn = data_loading_fn
        self.img_standardization_fn = img_standardization_fn
        self.pairing_fn = pairing_fn
        self.data_augmentation_fn = data_augmentation_fn
        self.noise_sigma = noise_sigma

        self.img_paths = _load_image_paths(data_paths, mode)

        random.shuffle(self.img_paths)
        if limit is not None:
            self.img_paths = self.img_paths[:limit]

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(
        self, idx: int
    ) -> tuple[npt.NDArray[np.uint8] | npt.NDArray[np.float32], npt.NDArray[np.uint8] | npt.NDArray[np.float32]]:
        clean_path = self.img_paths[idx % len(self.img_paths)]
        img_clean = self.data_loading_fn(clean_path)
        img_noisy = self.pairing_fn(clean_path)

        if self.data_augmentation_fn is not None:
            img_clean, img_noisy = self.data_augmentation_fn(img_clean, img_noisy)

        img_clean = self.img_standardization_fn(img_clean)
        img_noisy = self.img_standardization_fn(img_noisy)

        img_clean = hwc_to_chw(img_clean)
        img_noisy = hwc_to_chw(img_noisy)

        return img_clean, img_noisy


def _load_image_paths(data_paths: list[Path] | Path, mode: str) -> list[Path]:
    if isinstance(data_paths, Path):
        data_paths = [data_paths]

    img_paths: list[Path] = []

    for data_path in data_paths:
        indices_path = data_path / "indices" / f"{mode}_list.json"
        if not indices_path.exists():
            msg = f"Indices directory not found: {indices_path}. Please create train/val split first."
            raise FileNotFoundError(msg)

        with indices_path.open(encoding="utf-8") as f:
            files = json.load(f)
        img_paths += [Path(p) for p in files]

    return img_paths
