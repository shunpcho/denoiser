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

from denoiser.data.tiling import iter_image_tiles, pad_to_multiple_reflect
from denoiser.utils.alias import IMAGE_DIMENSIONS_3D, IndexMapEntry
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


class TiledPairedDataset(
    Dataset[
        tuple[
            npt.NDArray[np.uint8] | npt.NDArray[np.float32],
            npt.NDArray[np.uint8] | npt.NDArray[np.float32],
            IndexMapEntry,
        ]
    ]
):
    """Validation dataset that returns ALL tiles of each image.

    - Loads clean/noisy pair
    - Pads (reflect) to multiples of tile_size
    - Splits into tiles (grid, stride == tile_size)
    - Returns ONE tile per __getitem__ along with metadata for stitching.

    Returned:
        clean_tile_chw (float32), noisy_tile_chw (float32), meta (dict)
    """

    def __init__(
        self,
        data_paths: Path | list[Path],
        data_loading_fn: Callable[[Path], npt.NDArray[np.uint8]],
        img_standardization_fn: Callable[[npt.NDArray[np.uint8]], npt.NDArray[np.float32]],
        pairing_fn: Callable[[Path], npt.NDArray[np.uint8]],
        tile_size: int | tuple[int, int],
        mode: Literal["val"] = "val",
        limit: int | None = None,
    ) -> None:
        self.data_loading_fn = data_loading_fn
        self.img_standardization_fn = img_standardization_fn
        self.pairing_fn = pairing_fn

        self.img_paths = _load_image_paths(data_paths, mode)
        if limit is not None:
            self.img_paths = self.img_paths[:limit]

        if isinstance(tile_size, int):
            self.tile_h = self.tile_w = tile_size
        else:
            self.tile_h, self.tile_w = tile_size

        self._tile_fn = iter_image_tiles((self.tile_h, self.tile_w))

        self.index_map: list[IndexMapEntry] = []

        for img_i, p in enumerate(self.img_paths):
            img = self.data_loading_fn(p)
            orig_h, orig_w = img.shape[:2]
            padded, pad_h, pad_w = pad_to_multiple_reflect(img, self.tile_h, self.tile_w)
            padded_h, padded_w = padded.shape[:2]

            tiles_y = padded_h // self.tile_h
            tiles_x = padded_w // self.tile_w
            tiles_per_image = tiles_y * tiles_x

            tile_idx = 0
            for y in range(0, padded_h, self.tile_h):
                for x in range(0, padded_w, self.tile_w):
                    entry: IndexMapEntry = {
                        "img_i": img_i,
                        "tile_idx": tile_idx,
                        "tile_y": y,
                        "tile_x": x,
                        "tile_h": self.tile_h,
                        "tile_w": self.tile_w,
                        "tiles_per_image": tiles_per_image,
                        "orig_h": orig_h,
                        "orig_w": orig_w,
                        "padded_h": padded_h,
                        "padded_w": padded_w,
                        "pad_h": pad_h,
                        "pad_w": pad_w,
                        "image_id": str(p),  # path string (stable)
                    }
                    self.index_map.append(entry)
                    tile_idx += 1

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], IndexMapEntry]:
        m = self.index_map[idx]
        img_i = int(m["img_i"])
        tile_y = int(m["tile_y"])
        tile_x = int(m["tile_x"])
        tile_h = int(m["tile_h"])
        tile_w = int(m["tile_w"])

        clean_path = self.img_paths[img_i]

        clean, noisy = self._extract_and_standardize_tile(clean_path, tile_y, tile_x, tile_h, tile_w)

        meta = m
        return clean, noisy, meta

    def _extract_and_standardize_tile(
        self, clean_path: Path, tile_y: int, tile_x: int, tile_h: int, tile_w: int
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        clean_u8 = self.data_loading_fn(clean_path)
        noisy_u8 = self.pairing_fn(clean_path)

        # pad both the same way (reflect)
        clean_pad, _, _ = pad_to_multiple_reflect(clean_u8, tile_h, tile_w)
        noisy_pad, _, _ = pad_to_multiple_reflect(noisy_u8, tile_h, tile_w)

        # slice directly by coordinates (faster than iterating generator)
        if clean_pad.ndim == IMAGE_DIMENSIONS_3D:
            clean_tile_u8 = clean_pad[tile_y : tile_y + tile_h, tile_x : tile_x + tile_w, :]
            noisy_tile_u8 = noisy_pad[tile_y : tile_y + tile_h, tile_x : tile_x + tile_w, :]
        else:
            clean_tile_u8 = clean_pad[tile_y : tile_y + tile_h, tile_x : tile_x + tile_w]
            noisy_tile_u8 = noisy_pad[tile_y : tile_y + tile_h, tile_x : tile_x + tile_w]

        clean = hwc_to_chw(self.img_standardization_fn(clean_tile_u8))
        noisy = hwc_to_chw(self.img_standardization_fn(noisy_tile_u8))
        return clean, noisy
