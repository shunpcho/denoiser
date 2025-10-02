from __future__ import annotations

import numpy as np
import numpy.typing as npt


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
