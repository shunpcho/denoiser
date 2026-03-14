"""Tests for data loader and dataset classes."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from denoiser.configs.config import PairingKeyWords
from denoiser.data.data_loader import PairdDataset
from denoiser.data.data_transformations import pairing_clean_noisy


class TestGaussianNoiseFunction:
    """Test Gaussian noise function from pairing_clean_noisy."""

    def test_gaussian_noise_fn_creation(self) -> None:
        """Test that pairing_clean_noisy returns a callable for Gaussian noise."""
        sigma = 0.1
        noise_fn = pairing_clean_noisy(paring_words=None, noise_sigma=sigma)
        assert callable(noise_fn)

    def test_gaussian_noise_fn_application(self) -> None:
        """Test applying Gaussian noise function."""
        sigma = 0.1
        noise_fn = pairing_clean_noisy(paring_words=None, noise_sigma=sigma)

        # Create test image file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img = Image.new("RGB", (32, 32), color=(128, 128, 128))
            img.save(f.name)
            test_path = Path(f.name)

        try:
            noisy_img = noise_fn(test_path)
            assert isinstance(noisy_img, np.ndarray)
            assert noisy_img.shape == (32, 32, 3)
            assert noisy_img.dtype == np.uint8
        finally:
            test_path.unlink()

    @pytest.mark.parametrize("sigma", [0.05, 0.1, 0.2])
    def test_gaussian_noise_different_sigmas(self, sigma: float) -> None:
        """Test Gaussian noise with different sigma values."""
        noise_fn = pairing_clean_noisy(paring_words=None, noise_sigma=sigma)

        # Create test image file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img = Image.new("RGB", (16, 16), color=(100, 100, 100))
            img.save(f.name)
            test_path = Path(f.name)

        try:
            noisy_img = noise_fn(test_path)
            assert noisy_img.shape == (16, 16, 3)
            assert noisy_img.dtype == np.uint8
        finally:
            test_path.unlink()


class TestPairdDataset:
    """Test PairdDataset class."""

    @pytest.fixture
    def temp_image_dir(self) -> Path:
        """Create temporary directory with test images."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create test images
            for i in range(5):
                img = Image.new("RGB", (64, 64), color=(i * 50, i * 50, i * 50))
                img.save(tmp_path / f"test_image_{i}.png")

            yield tmp_path

    @pytest.fixture
    def mock_functions(self) -> dict:
        """Create mock functions for dataset initialization."""

        def mock_data_loading_fn(path: Path) -> np.ndarray:
            """Mock data loading function."""
            rng = np.random.default_rng(42)
            return rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)

        def mock_img_standardization_fn(img: np.ndarray) -> np.ndarray:
            """Mock image standardization function."""
            return img.astype(np.float32) / 255.0

        def mock_paring_fn(path: Path) -> np.ndarray:
            """Mock pairing function."""
            rng = np.random.default_rng(42)
            return rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)

        def mock_data_augmentation_fn(clean: np.ndarray, noisy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """Mock data augmentation function."""
            return clean, noisy

        return {
            "data_loading_fn": mock_data_loading_fn,
            "img_standardization_fn": mock_img_standardization_fn,
            "paring_fn": mock_paring_fn,
            "data_augmentation_fn": mock_data_augmentation_fn,
        }

    def test_dataset_initialization_single_path(self, temp_image_dir: Path, mock_functions: dict) -> None:
        """Test dataset initialization with single data path."""
        dataset = PairdDataset(
            data_paths=temp_image_dir,
            data_loading_fn=mock_functions["data_loading_fn"],
            img_standardization_fn=mock_functions["img_standardization_fn"],
            paring_fn=mock_functions["paring_fn"],
        )

        assert len(dataset) > 0
        assert hasattr(dataset, "img_paths")
        assert all(path.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tif"] for path in dataset.img_paths)

    def test_dataset_initialization_multiple_paths(self, temp_image_dir: Path, mock_functions: dict) -> None:
        """Test dataset initialization with multiple data paths."""
        dataset = PairdDataset(
            data_paths=[temp_image_dir, temp_image_dir],  # Same dir twice for testing
            data_loading_fn=mock_functions["data_loading_fn"],
            img_standardization_fn=mock_functions["img_standardization_fn"],
            paring_fn=mock_functions["paring_fn"],
        )

        assert len(dataset) > 0

    def test_dataset_with_limit(self, temp_image_dir: Path, mock_functions: dict) -> None:
        """Test dataset with limit parameter."""
        limit = 2
        dataset = PairdDataset(
            data_paths=temp_image_dir,
            data_loading_fn=mock_functions["data_loading_fn"],
            img_standardization_fn=mock_functions["img_standardization_fn"],
            paring_fn=mock_functions["paring_fn"],
            limit=limit,
        )

        assert len(dataset) == limit

    def test_dataset_getitem(self, temp_image_dir: Path, mock_functions: dict) -> None:
        """Test dataset __getitem__ method."""
        dataset = PairdDataset(
            data_paths=temp_image_dir,
            data_loading_fn=mock_functions["data_loading_fn"],
            img_standardization_fn=mock_functions["img_standardization_fn"],
            paring_fn=mock_functions["paring_fn"],
        )

        if len(dataset) > 0:
            clean, noisy = dataset[0]

            assert isinstance(clean, np.ndarray)
            assert isinstance(noisy, np.ndarray)
            assert clean.dtype == np.uint8
            assert noisy.dtype == np.uint8
            assert clean.shape == noisy.shape
            # Should be HWC format (PairdDataset returns raw images)
            assert len(clean.shape) == 3
            assert clean.shape[2] == 3  # RGB channels last

    def test_dataset_with_augmentation(self, temp_image_dir: Path, mock_functions: dict) -> None:
        """Test dataset with data augmentation."""
        dataset = PairdDataset(
            data_paths=temp_image_dir,
            data_loading_fn=mock_functions["data_loading_fn"],
            img_standardization_fn=mock_functions["img_standardization_fn"],
            paring_fn=mock_functions["paring_fn"],
            data_augmentation_fn=mock_functions["data_augmentation_fn"],
        )

        if len(dataset) > 0:
            clean, noisy = dataset[0]
            assert isinstance(clean, np.ndarray)
            assert isinstance(noisy, np.ndarray)

    def test_dataset_with_pairing_keywords(self, temp_image_dir: Path, mock_functions: dict) -> None:
        """Test dataset with pairing keywords."""
        pairing_keywords = PairingKeyWords(clean="clean", noisy="noisy")

        dataset = PairdDataset(
            data_paths=temp_image_dir,
            data_loading_fn=mock_functions["data_loading_fn"],
            img_standardization_fn=mock_functions["img_standardization_fn"],
            paring_fn=mock_functions["paring_fn"],
            img_read_keywards=pairing_keywords,
        )

        # Should still work even with no matching keywords in test images
        assert isinstance(dataset, PairdDataset)

    def test_dataset_len(self, temp_image_dir: Path, mock_functions: dict) -> None:
        """Test dataset __len__ method."""
        dataset = PairdDataset(
            data_paths=temp_image_dir,
            data_loading_fn=mock_functions["data_loading_fn"],
            img_standardization_fn=mock_functions["img_standardization_fn"],
            paring_fn=mock_functions["paring_fn"],
        )

        length = len(dataset)
        assert isinstance(length, int)
        assert length >= 0

    def test_dataset_empty_directory(self, mock_functions: dict) -> None:
        """Test dataset with empty directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            empty_dir = Path(tmp_dir)

            dataset = PairdDataset(
                data_paths=empty_dir,
                data_loading_fn=mock_functions["data_loading_fn"],
                img_standardization_fn=mock_functions["img_standardization_fn"],
                paring_fn=mock_functions["paring_fn"],
            )

            assert len(dataset) == 0

    def test_dataset_noise_sigma_parameter(self, temp_image_dir: Path, mock_functions: dict) -> None:
        """Test dataset with different noise_sigma values."""
        for sigma in [0.05, 0.1, 0.2]:
            dataset = PairdDataset(
                data_paths=temp_image_dir,
                data_loading_fn=mock_functions["data_loading_fn"],
                img_standardization_fn=mock_functions["img_standardization_fn"],
                paring_fn=mock_functions["paring_fn"],
                noise_sigma=sigma,
            )

            assert dataset.noise_sigma == sigma

    def test_dataset_index_wrapping(self, temp_image_dir: Path, mock_functions: dict) -> None:
        """Test that dataset handles index wrapping correctly."""
        dataset = PairdDataset(
            data_paths=temp_image_dir,
            data_loading_fn=mock_functions["data_loading_fn"],
            img_standardization_fn=mock_functions["img_standardization_fn"],
            paring_fn=mock_functions["paring_fn"],
        )

        if len(dataset) > 0:
            # Test normal indexing
            clean1, noisy1 = dataset[0]

            # Test wrapped indexing
            clean2, noisy2 = dataset[len(dataset)]  # Should wrap to index 0

            # Should get the same result due to modulo operation
            np.testing.assert_array_equal(clean1, clean2)
            np.testing.assert_array_equal(noisy1, noisy2)

    def test_dataset_sorted_paths(self, temp_image_dir: Path, mock_functions: dict) -> None:
        """Test that dataset sorts image paths."""
        dataset = PairdDataset(
            data_paths=temp_image_dir,
            data_loading_fn=mock_functions["data_loading_fn"],
            img_standardization_fn=mock_functions["img_standardization_fn"],
            paring_fn=mock_functions["paring_fn"],
        )

        # Check that paths are sorted
        paths = [str(p) for p in dataset.img_paths]
        assert paths == sorted(paths)
