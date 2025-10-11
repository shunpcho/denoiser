from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch


def collate_fn(
    batch: list[tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Custom collate function to handle numpy arrays and convert them to tensors.

    Args:
        batch: List of tuples (clean_img, noisy_img) where each image is a numpy array
               Images are expected to already be in CHW format from TrainSubset/ValSubset

    Returns:
        Tuple of batched tensors (clean_batch, noisy_batch)
    """
    clean_imgs = []
    noisy_imgs = []

    for clean_img, noisy_img in batch:
        # Convert numpy arrays to tensors
        clean_tensor = torch.from_numpy(clean_img).float()
        noisy_tensor = torch.from_numpy(noisy_img).float()

        clean_imgs.append(clean_tensor)
        noisy_imgs.append(noisy_tensor)

    # Stack tensors to create batches
    clean_batch = torch.stack(clean_imgs, dim=0)
    noisy_batch = torch.stack(noisy_imgs, dim=0)

    return clean_batch, noisy_batch


def hwc_to_chw(img: npt.NDArray[np.uint8] | npt.NDArray[np.float32]) -> npt.NDArray[np.uint8] | npt.NDArray[np.float32]:
    """Convert image from (H, W, C) to (C, H, W).

    Args:
        img (np.ndarray): Input image array of shape (H, W, C).

    Returns:
        np.ndarray: Transformed image array of shape (C, H, W).

    Raises:
        ValueError: If the input image does not have 3 dimensions.

    """
    if img.ndim != 3:  # Color dim
        msg = f"Expected 3D array (H, W, C), got shape {img.shape}"
        raise ValueError(msg)
    return np.transpose(img, (2, 0, 1))


def chw_to_hwc(img: npt.NDArray[np.uint8] | npt.NDArray[np.float32]) -> npt.NDArray[np.uint8] | npt.NDArray[np.float32]:
    """Convert image from (C, H, W) to (H, W, C).

    Args:
        img (np.ndarray): Input image array of shape (C, H, W).

    Returns:
        np.ndarray: Transformed image array of shape (H, W, C).

    Raises:
        ValueError: If the input image does not have 3 dimensions.

    """
    if img.ndim != 3:  # Color dim
        msg = f"Expected 3D array (C, H, W), got shape {img.shape}"
        raise ValueError(msg)
    return np.transpose(img, (1, 2, 0))
