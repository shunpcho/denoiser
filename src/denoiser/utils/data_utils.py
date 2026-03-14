from __future__ import annotations

from typing import Any, overload

import numpy as np
import numpy.typing as npt
import torch

IMAGE_DIMENSIONS_3D = 3  # Number of color channels (e.g., RGB)


class InvalidMetaTypeError(TypeError):
    def __init__(self, message: str) -> None:
        super().__init__(message)

    @staticmethod
    def build_message(meta: object) -> str:
        return f"meta must be dict, got: {type(meta)}"


BATCH_ENTRY_LENGTH_WITHOUT_META = 2


class InvalidBatchEntryLengthError(ValueError):
    def __init__(self, length: int) -> None:
        super().__init__(
            f"Expected {BATCH_ENTRY_LENGTH_WITHOUT_META} or "
            f"{BATCH_ENTRY_LENGTH_WITHOUT_META + 1} items per batch entry, got: {length}"
        )


def collate_fn(
    batch: list[tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]]
    | list[tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], dict[str, Any]]],
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    """Custom collate function.

    Supports:
      - batch item: (clean_img, noisy_img)
      - batch item: (clean_img, noisy_img, meta_dict)

    Returns:
      - (clean_batch, noisy_batch)
      - (clean_batch, noisy_batch, meta_list)

    Raises:
        InvalidBatchEntryLengthError: If a batch item does not have 2 or 3 elements.

    """
    clean_imgs: list[torch.Tensor] = []
    noisy_imgs: list[torch.Tensor] = []
    metas: list[dict[str, Any]] = []

    for item in batch:
        item_length = len(item)
        if item_length < BATCH_ENTRY_LENGTH_WITHOUT_META:
            raise InvalidBatchEntryLengthError(len(item))

        clean_img, noisy_img, *rest = item
        meta = rest[0] if rest else None

        clean_tensor = torch.from_numpy(clean_img).float()
        noisy_tensor = torch.from_numpy(noisy_img).float()

        clean_imgs.append(clean_tensor)
        noisy_imgs.append(noisy_tensor)

        if meta is not None:
            metas.append(meta)

    clean_batch = torch.stack(clean_imgs, dim=0)
    noisy_batch = torch.stack(noisy_imgs, dim=0)

    if metas:
        return clean_batch, noisy_batch, metas

    return clean_batch, noisy_batch


@overload
def hwc_to_chw(img: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]: ...


@overload
def hwc_to_chw(img: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]: ...


def hwc_to_chw(img: npt.NDArray[np.uint8] | npt.NDArray[np.float32]) -> npt.NDArray[np.uint8] | npt.NDArray[np.float32]:
    """Convert image from (H, W, C) to (C, H, W).

    Args:
        img (np.ndarray): Input image array of shape (H, W, C).

    Returns:
        np.ndarray: Transformed image array of shape (C, H, W).

    Raises:
        ValueError: If the input image does not have 3 dimensions.

    """
    if img.ndim != IMAGE_DIMENSIONS_3D:
        msg = f"Expected 3D array (H, W, C), got shape {img.shape}"
        raise ValueError(msg)
    return np.transpose(img, (2, 0, 1))  # type: ignore[return-value]


@overload
def chw_to_hwc(img: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]: ...


@overload
def chw_to_hwc(img: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]: ...


def chw_to_hwc(img: npt.NDArray[np.uint8] | npt.NDArray[np.float32]) -> npt.NDArray[np.uint8] | npt.NDArray[np.float32]:
    """Convert image from (C, H, W) to (H, W, C).

    Args:
        img (np.ndarray): Input image array of shape (C, H, W).

    Returns:
        np.ndarray: Transformed image array of shape (H, W, C).

    Raises:
        ValueError: If the input image does not have 3 dimensions.

    """
    if img.ndim != IMAGE_DIMENSIONS_3D:
        msg = f"Expected 3D array (C, H, W), got shape {img.shape}"
        raise ValueError(msg)
    return np.transpose(img, (1, 2, 0))  # type: ignore[return-value]
