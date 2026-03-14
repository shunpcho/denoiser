"""Tests for data transformation functions."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from denoiser.configs.config import PairingKeyWords
from denoiser.data.data_transformations import (
    compose_transformations,
    destandardize_img,
    load_img,
    load_img_clean,
    load_img_gray,
    paring_clean_noisy,
    random_crop,
    standardize_img,
)


class TestImageLoading:
    """Test image loading functions."""

    @pytest.fixture
    def sample_image_path(self) -> Path:
        """Create a temporary test image."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            # Create a simple test image
            img = Image.new("RGB", (64, 64), color=(128, 128, 128))
            img.save(f.name)
            return Path(f.name)

    @pytest.fixture
    def sample_gray_image_path(self) -> Path:
        """Create a temporary grayscale test image."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            # Create a simple grayscale test image
            img = Image.new("L", (64, 64), color=128)
            img.save(f.name)
            return Path(f.name)

    def test_load_img(self, sample_image_path: Path) -> None:
        """Test load_img function."""
        img = load_img(sample_image_path)

        assert isinstance(img, np.ndarray)
        assert img.dtype == np.uint8
        assert img.shape == (64, 64, 3)  # RGB image
        assert 0 <= img.min() <= img.max() <= 255

    def test_load_img_gray(self, sample_gray_image_path: Path) -> None:
        """Test load_img_gray function."""
        img = load_img_gray(sample_gray_image_path)

        assert isinstance(img, np.ndarray)
        assert img.dtype == np.uint8
        assert img.shape == (64, 64)  # Grayscale image (HxW format)
        assert 0 <= img.min() <= img.max() <= 255

    def test_load_img_clean_no_pairing(self) -> None:
        """Test load_img_clean with no pairing keywords."""
        load_fn = load_img_clean(None)

        # Should return the standard load_img function
        assert callable(load_fn)

    def test_load_img_clean_with_pairing(self) -> None:
        """Test load_img_clean with pairing keywords."""
        pairing_words = PairingKeyWords(clean="clean", noisy="noisy")
        load_fn = load_img_clean(pairing_words)

        assert callable(load_fn)


class TestImageStandardization:
    """Test image standardization functions."""

    @pytest.fixture
    def sample_image(self) -> np.ndarray:
        """Create a sample test image."""
        return np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)

    def test_standardize_img_default(self, sample_image: np.ndarray) -> None:
        """Test standardize_img with default parameters."""
        standardize_fn = standardize_img()
        result = standardize_fn(sample_image)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == sample_image.shape

        # Should normalize to [0, 1] by default
        assert 0.0 <= result.min() <= result.max() <= 1.0

    def test_standardize_img_custom_params(self, sample_image: np.ndarray) -> None:
        """Test standardize_img with custom mean and std."""
        mean = (0.5, 0.5, 0.5)
        std = (0.25, 0.25, 0.25)
        standardize_fn = standardize_img(mean=mean, std=std)
        result = standardize_fn(sample_image)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == sample_image.shape

    def test_destandardize_img(self) -> None:
        """Test destandardize_img function."""
        # Create normalized image
        normalized_img = np.random.rand(32, 32, 3).astype(np.float32)

        destandardize_fn = destandardize_img()
        result = destandardize_fn(normalized_img)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8
        assert result.shape == normalized_img.shape
        assert 0 <= result.min() <= result.max() <= 255

    def test_standardize_destandardize_roundtrip(self, sample_image: np.ndarray) -> None:
        """Test that standardize -> destandardize recovers original image."""
        standardize_fn = standardize_img()
        destandardize_fn = destandardize_img()

        # Forward pass
        standardized = standardize_fn(sample_image)
        recovered = destandardize_fn(standardized)

        # Should recover original image with small error due to float conversion
        assert np.allclose(sample_image, recovered, atol=1)


class TestPairing:
    """Test image pairing functions."""

    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create a temporary directory with test images."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create clean and noisy test images
            for i in range(3):
                clean_img = Image.new("RGB", (64, 64), color=(100, 100, 100))
                noisy_img = Image.new("RGB", (64, 64), color=(120, 120, 120))

                clean_img.save(tmp_path / f"image_{i}_clean.png")
                noisy_img.save(tmp_path / f"image_{i}_noisy.png")

            yield tmp_path

    def test_paring_gaussian_noise(self) -> None:
        """Test pairing with Gaussian noise generation."""
        noise_sigma = 0.1
        pairing_fn = paring_clean_noisy(None, noise_sigma)

        assert callable(pairing_fn)

    def test_paring_existing_noisy_images(self, temp_dir: Path) -> None:
        """Test pairing with existing noisy images."""
        pairing_words = PairingKeyWords(clean="clean", noisy="noisy")
        pairing_fn = paring_clean_noisy(pairing_words, 0.0)

        clean_path = temp_dir / "image_0_clean.png"
        noisy_img = pairing_fn(clean_path)

        assert isinstance(noisy_img, np.ndarray)
        assert noisy_img.dtype == np.uint8
        assert noisy_img.shape == (64, 64, 3)


class TestDataTransformations:
    """Test data transformation utilities."""

    @pytest.fixture
    def sample_image_pair(self) -> tuple[np.ndarray, np.ndarray]:
        """Create sample clean and noisy image pair."""
        clean = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
        noisy = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
        return clean, noisy

    def test_random_crop(self, sample_image_pair: tuple[np.ndarray, np.ndarray]) -> None:
        """Test random_crop function."""
        clean, noisy = sample_image_pair
        crop_size = 32

        crop_fn = random_crop(crop_size)
        clean_crop, noisy_crop = crop_fn(clean, noisy)

        assert clean_crop.shape == (crop_size, crop_size, 3)
        assert noisy_crop.shape == (crop_size, crop_size, 3)
        assert clean_crop.dtype == clean.dtype
        assert noisy_crop.dtype == noisy.dtype

    def test_random_crop_larger_than_image(self, sample_image_pair: tuple[np.ndarray, np.ndarray]) -> None:
        """Test random_crop with crop size larger than image."""
        clean, noisy = sample_image_pair
        crop_size = 128  # Larger than 64x64 image

        crop_fn = random_crop(crop_size)

        # Should handle gracefully or raise appropriate error
        with pytest.raises((ValueError, IndexError)):
            crop_fn(clean, noisy)

    def test_compose_transformations_empty(self, sample_image_pair: tuple[np.ndarray, np.ndarray]) -> None:
        """Test compose_transformations with empty transform list."""
        clean, noisy = sample_image_pair

        compose_fn = compose_transformations([])
        result_clean, result_noisy = compose_fn(clean, noisy)

        np.testing.assert_array_equal(result_clean, clean)
        np.testing.assert_array_equal(result_noisy, noisy)

    def test_compose_transformations_single(self, sample_image_pair: tuple[np.ndarray, np.ndarray]) -> None:
        """Test compose_transformations with single transformation."""
        clean, noisy = sample_image_pair
        crop_size = 32

        transforms = [random_crop(crop_size)]
        compose_fn = compose_transformations(transforms)
        result_clean, result_noisy = compose_fn(clean, noisy)

        assert result_clean.shape == (crop_size, crop_size, 3)
        assert result_noisy.shape == (crop_size, crop_size, 3)

    def test_compose_transformations_multiple(self, sample_image_pair: tuple[np.ndarray, np.ndarray]) -> None:
        """Test compose_transformations with multiple transformations."""
        clean, noisy = sample_image_pair

        # Identity transform for testing
        def identity_transform(img_clean: np.ndarray, img_noisy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return img_clean, img_noisy

        transforms = [identity_transform, identity_transform]
        compose_fn = compose_transformations(transforms)
        result_clean, result_noisy = compose_fn(clean, noisy)

        np.testing.assert_array_equal(result_clean, clean)
        np.testing.assert_array_equal(result_noisy, noisy)


class TestConstants:
    """Test module constants."""

    def test_constants_exist(self) -> None:
        """Test that required constants are defined."""
        from denoiser.data.data_transformations import MAX_PIXEL_VALUE, MIN_PIXEL_VALUE, RGB_CHANNELS

        assert MIN_PIXEL_VALUE == 0
        assert MAX_PIXEL_VALUE == 255
        assert RGB_CHANNELS == 3
