from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from denoiser.configs.config import PairingKeyWords


# Constants
MIN_PIXEL_VALUE = 0
MAX_PIXEL_VALUE = 255
MIN_DETECTOR_COUNT = 2
RGB_CHANNELS = 3
IMAGE_DIMENSIONS_3D = 3  # (H, W, C)


# data_loading_fn
def load_img(path: Path) -> npt.NDArray[np.uint8]:
    """Load image as RGB and convert numpy."""
    img = Image.open(path).convert("RGB")
    # Ensure proper conversion to avoid dtype issues
    img_array = np.asarray(img)
    return img_array.astype(np.uint8)


def load_img_gray(path: Path) -> npt.NDArray[np.uint8]:
    """Load image as grayscale and convert numpy."""
    img = Image.open(path).convert("L")
    # Ensure proper conversion to avoid dtype issues
    img_array = np.asarray(img)
    return img_array.astype(np.uint8)


def load_img_clean(paring_words: PairingKeyWords | None) -> Callable[[Path], npt.NDArray[np.uint8]]:
    """Return function to get clean image loading function based on pairing configuration."""
    if paring_words is None:
        # Multi-detector case
        if len(paring_words.detector) >= MIN_DETECTOR_COUNT:

            def load_dual_detector(path: Path) -> npt.NDArray[np.uint8]:
                """Load two detector images and concatenate channel-wise."""
                detect_a_path = path
                detect_b_path = path.with_name(path.name.replace(paring_words.detector[0], paring_words.detector[1]))
                img_detect_a = load_img_gray(detect_a_path)
                img_detect_b = load_img_gray(detect_b_path)
                # Expand dimensions: (H, W) -> (H, W, 1)
                img_a = np.expand_dims(img_detect_a, axis=-1)
                img_b = np.expand_dims(img_detect_b, axis=-1)
                # Concatenate: (H, W, 2)
                return np.concatenate([img_a, img_b], axis=-1)

            return load_dual_detector

        # Single detector case
        elif len(paring_words.detector) == 1:

            def load_single_detector(path: Path) -> npt.NDArray[np.uint8]:
                """Load single detector image as grayscale with channel dimension."""
                img_detect = load_img_gray(path)
                # Add channel dimension: (H, W) -> (H, W, 1)
                return np.expand_dims(img_detect, axis=-1)

            return load_single_detector

    # Default case: simple RGB loading
    return load_img


# Pairing functions
def paring_clean_noisy(
    paring_words: PairingKeyWords | None, noise_sigma: float
) -> Callable[[Path], npt.NDArray[np.uint8]]:
    """Create function to load noisy images from clean image paths.

    Args:
        paring_words: Dictionary containing pairing keywords for clean/noisy images
        noise_sigma: Standard deviation for Gaussian noise when generating synthetic noise

    Returns:
        Function that takes a clean image path and returns noisy image array
    """
    # Case 1: Generate synthetic Gaussian noise
    if paring_words is None:

        def add_gaussian_noise(path: Path) -> npt.NDArray[np.uint8]:
            """Add Gaussian noise to clean image."""
            img = Image.open(path).convert("RGB")
            img_array = np.asarray(img).astype(np.uint8)
            rng = np.random.default_rng()
            noise = rng.normal(0, noise_sigma, img_array.shape).astype(np.int16)
            noisy_img = img_array.astype(np.int16) + noise
            return np.clip(noisy_img, MIN_PIXEL_VALUE, MAX_PIXEL_VALUE).astype(np.uint8)

        return add_gaussian_noise

    # Case 2: Load existing noisy images (RGB)
    elif paring_words.noisy is not None and paring_words.detector is None:

        def load_noisy_rgb(path: Path) -> npt.NDArray[np.uint8]:
            """Load corresponding noisy image as RGB."""
            noisy_path = Path(str(path).replace(paring_words.clean, paring_words.noisy))  # type: ignore[union-attr]
            img = Image.open(noisy_path).convert("RGB")
            img_array = np.asarray(img)
            return img_array.astype(np.uint8)

        return load_noisy_rgb

    # Case 3: Load detector images
    if paring_words.detector is not None and len(paring_words.detector) >= MIN_DETECTOR_COUNT:

        def load_noisy_dual_detector(path: Path) -> npt.NDArray[np.uint8]:
            """Load dual detector noisy images and concatenate channel-wise."""
            noisy_path = Path(str(path).replace(paring_words.clean, paring_words.noisy))  # type: ignore[union-attr]
            detect_a_path = noisy_path
            detect_b_path = noisy_path.with_name(
                noisy_path.name.replace(paring_words.detector[0], paring_words.detector[1])
            )
            img_detect_a = load_img_gray(detect_a_path)
            img_detect_b = load_img_gray(detect_b_path)
            # Expand dimensions: (H, W) -> (H, W, 1)
            img_a = np.expand_dims(img_detect_a, axis=-1)
            img_b = np.expand_dims(img_detect_b, axis=-1)
            # Concatenate: (H, W, 2)
            return np.concatenate([img_a, img_b], axis=-1)

        return load_noisy_dual_detector

    elif len(paring_words.detector) == 1:

        def load_noisy_single_detector(path: Path) -> npt.NDArray[np.uint8]:
            """Load single detector noisy image as grayscale with channel dimension."""
            noisy_path = Path(str(path).replace(paring_words.clean, paring_words.noisy))  # type: ignore[union-attr]
            img_detect = load_img_gray(noisy_path)
            # Add channel dimension: (H, W) -> (H, W, 1)
            return np.expand_dims(img_detect, axis=-1)

        return load_noisy_single_detector

    # Fallback: load noisy RGB image
    def load_noisy_rgb_fallback(path: Path) -> npt.NDArray[np.uint8]:
        """Fallback: Load noisy RGB image."""
        noisy_path = Path(str(path).replace(paring_words.clean, paring_words.noisy))  # type: ignore[union-attr]
        img = Image.open(noisy_path).convert("RGB")
        img_array = np.asarray(img)
        return img_array.astype(np.uint8)

    return load_noisy_rgb_fallback


def compose_transformations(
    transforms: list[
        Callable[
            [npt.NDArray[np.uint8 | np.float32], npt.NDArray[np.uint8 | np.float32]],
            tuple[npt.NDArray[np.uint8 | np.float32], npt.NDArray[np.uint8 | np.float32]],
        ]
    ],
) -> Callable[
    [npt.NDArray[np.uint8 | np.float32], npt.NDArray[np.uint8 | np.float32]],
    tuple[npt.NDArray[np.uint8 | np.float32], npt.NDArray[np.uint8 | np.float32]],
]:
    """Compose multiple transformations into one."""

    def _fn(
        img_clean: npt.NDArray[np.uint8 | np.float32],
        img_noisy: npt.NDArray[np.uint8 | np.float32],
    ) -> tuple[npt.NDArray[np.uint8 | np.float32], npt.NDArray[np.uint8 | np.float32]]:
        for transform in transforms:
            img_clean, img_noisy = transform(img_clean, img_noisy)
        return img_clean, img_noisy

    return _fn


# img_standardization_fn
def standardize_img(
    mean: float | tuple[float, float] | tuple[float, float, float],
    std: float | tuple[float, float] | tuple[float, float, float],
) -> Callable[[npt.NDArray[np.uint8]], npt.NDArray[np.float32]]:
    """Create function to standardize image.

    Args:
        mean: Mean for each channel.
        std: Std for each channel.

    Returns:
        Function to standardize image.
    """
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    max_val = 255.0

    def standardize(img: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        # Convert uint8 to float32 first to avoid type mismatch
        img = img.astype(np.float32)
        return (img - mean * max_val) / (std * max_val)

    return standardize


def destandardize_img(
    mean: float | tuple[float, float] | tuple[float, float, float],
    std: float | tuple[float, float] | tuple[float, float, float],
) -> Callable[[npt.NDArray[np.float32] | torch.Tensor], npt.NDArray[np.uint8]]:
    """Create function to destandardize image.

    Args:
        mean: Mean for each channel.
        std: Std for each channel.

    Returns:
        Function to destandardize image.
    """
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    max_val = 255.0

    def destandardize(img: npt.NDArray[np.float32] | torch.Tensor) -> npt.NDArray[np.uint8]:
        if isinstance(img, torch.Tensor):
            # (C, H, W) -> (H, W, C) for single image, or (N, C, H, W) -> (N, H, W, C) for batch
            batch_dims = 4
            img = img.permute(0, 2, 3, 1) if img.dim() == batch_dims else img.permute(1, 2, 0)
            img = img.cpu().detach().numpy()
        img = (img * std + mean) * max_val
        img = np.clip(img, a_min=0, a_max=255).astype(np.uint8)
        return img

    return destandardize


def destandardize_tensor(
    img_mean: tuple[float, ...], img_std: tuple[float, ...]
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create function to destandardize tensor image.

    Args:
        img_mean: Mean for each channel.
        img_std: Std for each channel.

    Returns:
        Function to destandardize tensor image.
    """
    mean = torch.Tensor(img_mean)
    std = torch.Tensor(img_std)

    def destandardize(img: torch.Tensor) -> torch.Tensor:
        mean_tensor = mean.to(img.device)
        std_tensor = std.to(img.device)
        # Destandardize: (img * std + mean) * 255
        img = (img * std_tensor.view(1, -1, 1, 1) + mean_tensor.view(1, -1, 1, 1)) * 255
        img = torch.clamp(img, min=0, max=255).type(torch.uint8)
        return img

    return destandardize


# Augmentation functions
def random_crop(
    crop_size: int | tuple[int, int] | None,
) -> Callable[[npt.NDArray[np.uint8], npt.NDArray[np.uint8]], tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]]:
    """Create function to perform random crop on image.

    Args:
        crop_size: Size of the crop. If int, creates square crop. If tuple, (height, width).
                  If None, returns original images without cropping.

    Returns:
        Function to perform random crop on image.
    """
    # If no crop size specified, return identity function
    if crop_size is None:

        def no_crop(
            img_clean: npt.NDArray[np.uint8], img_noisy: npt.NDArray[np.uint8]
        ) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
            return img_clean, img_noisy

        return no_crop

    if isinstance(crop_size, int):
        crop_h = crop_w = crop_size
    else:
        crop_h, crop_w = crop_size

    def rnd_crop(
        img_clean: npt.NDArray[np.uint8], img_noisy: npt.NDArray[np.uint8]
    ) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
        h, w = img_clean.shape[:2]

        # If image is smaller than crop size, return original image
        if h < crop_h or w < crop_w:
            return img_clean, img_noisy

        # Generate random top-left corner
        rng = np.random.default_rng()
        top = rng.integers(0, h - crop_h + 1)
        left = rng.integers(0, w - crop_w + 1)

        # Perform crop
        if len(img_clean.shape) == IMAGE_DIMENSIONS_3D:  # RGB or multi-channel (H, W, C)
            return img_clean[top : top + crop_h, left : left + crop_w, :], img_noisy[
                top : top + crop_h, left : left + crop_w, :
            ]
        else:  # Grayscale (H, W)
            return img_clean[top : top + crop_h, left : left + crop_w], img_noisy[
                top : top + crop_h, left : left + crop_w
            ]

    return rnd_crop


def specific_crop(
    crop_size: int | tuple[int, int],
) -> Callable[[npt.NDArray[np.uint8]], Generator[npt.NDArray[np.uint8], None, None]]:
    """Create function to perform specific crop on image.

    Args:
        crop_size: Size of the crop. If int, creates square crop. If tuple, (height, width).

    Returns:
        Function to perform specific crop on image.
    """
    if isinstance(crop_size, int):
        crop_h = crop_w = crop_size
    else:
        crop_h, crop_w = crop_size

    def spec_crop(img: npt.NDArray[np.uint8]) -> Generator[npt.NDArray[np.uint8], None, None]:
        h, w = img.shape[:2]

        for crop_y in range(0, h, crop_h):
            for crop_x in range(0, w, crop_w):
                if len(img.shape) == RGB_CHANNELS:  # RGB or multi-channel
                    img_crop = img[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w, :]
                else:  # Grayscale
                    img_crop = img[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
                yield img_crop

    return spec_crop
