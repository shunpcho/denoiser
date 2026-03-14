import sys
from pathlib import Path

import numpy as np

# Ensure package source is importable when tests run from workspace root
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "denoiser" / "src"))

from denoiser.data.data_transformations import (
    compose_transformations,
    destandardize_img,
    random_crop,
    standardize_img,
)


def test_standardize_destandardize_roundtrip():
    # standardize expects image as (H, W, C)
    img = np.array([[[0, 0, 255], [127, 0, 255]], [[255, 0, 255], [64, 0, 255]]], dtype=np.uint8)
    mean = (0.0, 0.0, 0.0)
    std = (1.0, 1.0, 1.0)
    std_fn = standardize_img(mean, std)
    dest_fn = destandardize_img(mean, std)
    s = std_fn(img)
    r = dest_fn(s)
    assert r.shape == img.shape
    assert np.allclose(r.astype(np.float32), img.astype(np.float32), atol=1.0)


def test_random_crop_and_compose():
    # Use (H, W, C) layout and pass both clean and noisy images
    img = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    crop_fn = random_crop(4)
    c_clean, c_noisy = crop_fn(img, img)
    assert c_clean.shape[0] == 4 and c_clean.shape[1] == 4
    composed = compose_transformations([crop_fn])
    c2_clean, c2_noisy = composed(img, img)
    assert c2_clean.shape[0] == 4 and c2_clean.shape[1] == 4
