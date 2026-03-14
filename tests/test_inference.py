"""Tests for inference functionality."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from denoiser.inference import batch_denoise, denoise_image, load_model, postprocess_image, preprocess_image
from denoiser.models.unet import UNet


class TestPreprocessImage:
    """Test preprocess_image function."""

    @pytest.fixture
    def sample_image_path(self) -> Path:
        """Create a temporary test image."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img = Image.new("RGB", (128, 128), color=(100, 150, 200))
            img.save(f.name)
            return Path(f.name)

    @pytest.fixture
    def mock_standardize_fn(self) -> callable:
        """Create a mock standardization function."""

        def standardize(img: np.ndarray) -> np.ndarray:
            return img.astype(np.float32) / 255.0

        return standardize

    def test_preprocess_image_cpu(self, sample_image_path: Path, mock_standardize_fn: callable) -> None:
        """Test image preprocessing on CPU."""
        device = torch.device("cpu")
        tensor = preprocess_image(sample_image_path, device, mock_standardize_fn)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.device.type == "cpu"
        assert tensor.dtype == torch.float32
        assert tensor.shape[0] == 1  # Batch dimension
        assert tensor.shape[1] == 3  # RGB channels
        assert len(tensor.shape) == 4  # Batch, Channel, Height, Width

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_preprocess_image_gpu(self, sample_image_path: Path, mock_standardize_fn: callable) -> None:
        """Test image preprocessing on GPU."""
        device = torch.device("cuda")
        tensor = preprocess_image(sample_image_path, device, mock_standardize_fn)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.device.type == "cuda"
        assert tensor.dtype == torch.float32

    def test_preprocess_image_values(self, sample_image_path: Path, mock_standardize_fn: callable) -> None:
        """Test that preprocessing produces expected value ranges."""
        device = torch.device("cpu")
        tensor = preprocess_image(sample_image_path, device, mock_standardize_fn)

        # Values should be normalized to [0, 1] by mock function
        assert 0.0 <= tensor.min().item() <= tensor.max().item() <= 1.0


class TestPostprocessImage:
    """Test postprocess_image function."""

    @pytest.fixture
    def mock_destandardize_fn(self) -> callable:
        """Create a mock destandardization function."""

        def destandardize(img: np.ndarray) -> np.ndarray:
            return (img * 255.0).astype(np.uint8)

        return destandardize

    def test_postprocess_image_shape(self, mock_destandardize_fn: callable) -> None:
        """Test postprocessing preserves spatial dimensions."""
        # Create tensor in NCHW format
        tensor = torch.rand(1, 3, 64, 64)
        result = postprocess_image(tensor, mock_destandardize_fn)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8
        assert result.shape == (64, 64, 3)  # HWC format

    def test_postprocess_image_values(self, mock_destandardize_fn: callable) -> None:
        """Test postprocessing produces valid uint8 values."""
        tensor = torch.rand(1, 3, 32, 32)
        result = postprocess_image(tensor, mock_destandardize_fn)

        assert 0 <= result.min() <= result.max() <= 255
        assert result.dtype == np.uint8

    def test_postprocess_batch_tensor(self, mock_destandardize_fn: callable) -> None:
        """Test postprocessing handles batch dimension correctly."""
        # Should only process first item in batch
        tensor = torch.rand(2, 3, 32, 32)
        result = postprocess_image(tensor, mock_destandardize_fn)

        assert result.shape == (32, 32, 3)


class TestLoadModel:
    """Test load_model function."""

    @pytest.fixture
    def mock_model_path(self) -> Path:
        """Create a temporary model file."""
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            # Create and save a simple model
            model = UNet(in_ch=3, out_ch=3, base_ch=32)
            torch.save(model.state_dict(), f.name)
            return Path(f.name)

    def test_load_model_cpu(self, mock_model_path: Path) -> None:
        """Test loading model on CPU."""
        device = torch.device("cpu")
        model = load_model(mock_model_path, device)

        assert isinstance(model, UNet)
        assert next(model.parameters()).device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_load_model_gpu(self, mock_model_path: Path) -> None:
        """Test loading model on GPU."""
        device = torch.device("cuda")
        model = load_model(mock_model_path, device)

        assert isinstance(model, UNet)
        assert next(model.parameters()).device.type == "cuda"

    def test_load_model_custom_architecture(self, mock_model_path: Path) -> None:
        """Test loading model with custom architecture parameters."""
        device = torch.device("cpu")
        model = load_model(mock_model_path, device, in_ch=1, out_ch=1, base_ch=16)

        assert isinstance(model, UNet)
        # Note: The loaded weights might not match the custom architecture,
        # but the function should still create the model with specified parameters

    def test_load_model_nonexistent_file(self) -> None:
        """Test loading model from non-existent file."""
        device = torch.device("cpu")
        nonexistent_path = Path("nonexistent_model.pth")

        with pytest.raises(FileNotFoundError):
            load_model(nonexistent_path, device)


class TestDenoiseImage:
    """Test denoise_image function."""

    @pytest.fixture
    def mock_model(self) -> UNet:
        """Create a mock denoising model."""
        model = UNet(in_ch=3, out_ch=3, base_ch=32)
        model.eval()
        return model

    @pytest.fixture
    def sample_image_path(self) -> Path:
        """Create a temporary test image."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img = Image.new("RGB", (64, 64), color=(100, 100, 100))
            img.save(f.name)
            return Path(f.name)

    def test_denoise_image_cpu(self, mock_model: UNet, sample_image_path: Path) -> None:
        """Test image denoising on CPU."""
        device = torch.device("cpu")
        mock_model = mock_model.to(device)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as output_file:
            output_path = Path(output_file.name)

        # Should not raise an error
        denoise_image(mock_model, sample_image_path, output_path, device)

        # Output file should exist
        assert output_path.exists()

        # Should be a valid image
        img = Image.open(output_path)
        assert img.size == (64, 64)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_denoise_image_gpu(self, mock_model: UNet, sample_image_path: Path) -> None:
        """Test image denoising on GPU."""
        device = torch.device("cuda")
        mock_model = mock_model.to(device)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as output_file:
            output_path = Path(output_file.name)

        denoise_image(mock_model, sample_image_path, output_path, device)
        assert output_path.exists()

    def test_denoise_image_creates_output_directory(self, mock_model: UNet, sample_image_path: Path) -> None:
        """Test that denoising creates output directory if it doesn't exist."""
        device = torch.device("cpu")

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "subdir" / "output.png"

            denoise_image(mock_model, sample_image_path, output_path, device)

            assert output_path.exists()
            assert output_path.parent.exists()


class TestBatchDenoise:
    """Test batch_denoise function."""

    @pytest.fixture
    def mock_model(self) -> UNet:
        """Create a mock denoising model."""
        model = UNet(in_ch=3, out_ch=3, base_ch=32)
        model.eval()
        return model

    @pytest.fixture
    def input_dir_with_images(self) -> Path:
        """Create a directory with multiple test images."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "input"
            input_path.mkdir()

            # Create test images
            for i, ext in enumerate([".jpg", ".png", ".jpeg"]):
                img = Image.new("RGB", (32, 32), color=(i * 50, i * 50, i * 50))
                img.save(input_path / f"test_{i}{ext}")

            yield input_path

    def test_batch_denoise_basic(self, mock_model: UNet, input_dir_with_images: Path) -> None:
        """Test basic batch denoising functionality."""
        device = torch.device("cpu")

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "output"

            batch_denoise(mock_model, input_dir_with_images, output_path, device)

            # Output directory should be created
            assert output_path.exists()

            # Should have processed files
            output_files = list(output_path.glob("*"))
            input_files = list(input_dir_with_images.glob("*"))

            # Should have same number of output files as valid input images
            valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
            valid_input_files = [f for f in input_files if f.suffix.lower() in valid_extensions]
            assert len(output_files) == len(valid_input_files)

    def test_batch_denoise_empty_input_dir(self, mock_model: UNet) -> None:
        """Test batch denoising with empty input directory."""
        device = torch.device("cpu")

        with tempfile.TemporaryDirectory() as tmp_dir:
            empty_input = Path(tmp_dir) / "empty_input"
            empty_input.mkdir()
            output_path = Path(tmp_dir) / "output"

            batch_denoise(mock_model, empty_input, output_path, device)

            # Output directory should still be created
            assert output_path.exists()

            # Should have no output files
            output_files = list(output_path.glob("*"))
            assert len(output_files) == 0

    def test_batch_denoise_custom_extensions(self, mock_model: UNet) -> None:
        """Test batch denoising with custom image extensions."""
        device = torch.device("cpu")

        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "input"
            input_path.mkdir()
            output_path = Path(tmp_dir) / "output"

            # Create image with custom extension
            img = Image.new("RGB", (32, 32), color=(100, 100, 100))
            img.save(input_path / "test.bmp")

            # Test with custom extensions
            batch_denoise(mock_model, input_path, output_path, device, image_extensions=(".bmp",))

            output_files = list(output_path.glob("*"))
            assert len(output_files) == 1

    def test_batch_denoise_preserves_filenames(self, mock_model: UNet, input_dir_with_images: Path) -> None:
        """Test that batch denoising preserves input filenames."""
        device = torch.device("cpu")

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "output"

            batch_denoise(mock_model, input_dir_with_images, output_path, device)

            input_stems = {
                f.stem for f in input_dir_with_images.glob("*") if f.suffix.lower() in {".jpg", ".png", ".jpeg"}
            }
            output_stems = {f.stem for f in output_path.glob("*")}

            # Should preserve base filenames
            assert input_stems == output_stems


class TestInferenceIntegration:
    """Integration tests for inference functionality."""

    @staticmethod
    def test_full_inference_pipeline() -> None:
        """Test complete inference pipeline from model creation to output."""
        device = torch.device("cpu")

        # Create model
        model = UNet(in_ch=3, out_ch=3, base_ch=16)  # Small model for testing
        model.eval()

        # Create test image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as input_file:
            img = Image.new("RGB", (64, 64), color=(128, 128, 128))
            img.save(input_file.name)
            input_path = Path(input_file.name)

        # Run inference
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as output_file:
            output_path = Path(output_file.name)

        denoise_image(model, input_path, output_path, device)

        # Verify output
        assert output_path.exists()
        output_img = Image.open(output_path)
        assert output_img.size == (64, 64)
        assert output_img.mode == "RGB"

    @staticmethod
    def test_model_inference_deterministic() -> None:
        """Test that inference produces deterministic results."""
        device = torch.device("cpu")
        torch.manual_seed(42)

        model = UNet(in_ch=3, out_ch=3, base_ch=16)
        model.eval()

        # Create test input
        with torch.no_grad():
            input_tensor = torch.rand(1, 3, 32, 32)

            # Run inference twice
            output1 = model(input_tensor)
            output2 = model(input_tensor)

            # Should be identical
            assert torch.allclose(output1, output2)
