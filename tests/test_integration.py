"""Integration tests for the denoiser package."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
from PIL import Image

from denoiser.configs.config import TrainConfig
from denoiser.data.data_loader import PairdDataset
from denoiser.data.data_transformations import load_img, paring_clean_noisy, standardize_img
from denoiser.inference import denoise_image, load_model
from denoiser.models.unet import UNet
from denoiser.utils.calculate_loss import denoiser_loss


class TestFullPipeline:
    """Integration tests for complete denoiser pipeline."""

    @pytest.fixture
    def sample_config(self) -> TrainConfig:
        """Create sample training configuration."""
        return TrainConfig(
            batch_size=2,
            crop_size=32,
            learning_rate=1e-4,
            iteration=10,
            interval=5,
        )

    @pytest.fixture
    def test_images_dir(self) -> Path:
        """Create directory with test images."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            img_dir = Path(tmp_dir) / "images"
            img_dir.mkdir()

            # Create test images
            for i in range(3):
                img = Image.new("RGB", (64, 64), color=(50 + i * 50, 100, 150))
                img.save(img_dir / f"test_{i}.png")

            yield img_dir

    @pytest.mark.integration
    def test_model_creation_and_forward_pass(self, sample_config: TrainConfig) -> None:
        """Test model creation and basic forward pass."""
        # Create model
        model = UNet(in_ch=3, out_ch=3, base_ch=16)

        # Create input tensor
        device = torch.device("cpu")
        input_tensor = torch.rand(sample_config.batch_size, 3, sample_config.crop_size, sample_config.crop_size)

        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)

        assert output.shape == input_tensor.shape
        assert not torch.isnan(output).any()

    @pytest.mark.integration
    def test_dataset_creation_and_loading(self, test_images_dir: Path, sample_config: TrainConfig) -> None:
        """Test dataset creation and data loading."""
        # Create dataset
        dataset = PairdDataset(
            data_paths=test_images_dir,
            data_loading_fn=load_img,
            img_standardization_fn=standardize_img(),
            paring_fn=paring_clean_noisy(None, 0.1),
            limit=2,
        )

        assert len(dataset) == 2

        # Load sample
        clean, noisy = dataset[0]
        assert clean.shape == noisy.shape
        assert clean.shape[0] == 3  # RGB channels
        assert len(clean.shape) == 3  # CHW format

    @pytest.mark.integration
    def test_loss_computation_with_model(self, sample_config: TrainConfig) -> None:
        """Test loss computation with actual model."""
        # Create model and data
        model = UNet(in_ch=3, out_ch=3, base_ch=16)
        device = torch.device("cpu")

        clean = torch.rand(sample_config.batch_size, 3, sample_config.crop_size, sample_config.crop_size)
        noisy = torch.rand(sample_config.batch_size, 3, sample_config.crop_size, sample_config.crop_size)

        # Compute loss
        losses = denoiser_loss(model, clean, noisy)

        assert "Loss" in losses
        assert "PSNR" in losses
        assert "SSIM" in losses
        assert losses["Loss"].item() >= 0
        assert losses["PSNR"].item() >= 0
        assert 0 <= losses["SSIM"].item() <= 1

    @pytest.mark.integration
    def test_model_save_and_load_cycle(self) -> None:
        """Test saving and loading model."""
        # Create model
        original_model = UNet(in_ch=3, out_ch=3, base_ch=16)
        device = torch.device("cpu")

        # Save model
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            model_path = Path(f.name)
            torch.save(original_model.state_dict(), model_path)

        try:
            # Load model
            loaded_model = load_model(model_path, device)

            # Test that loaded model works
            test_input = torch.rand(1, 3, 64, 64)

            original_model.eval()
            loaded_model.eval()

            with torch.no_grad():
                original_output = original_model(test_input)
                loaded_output = loaded_model(test_input)

            # Outputs should be identical
            assert torch.allclose(original_output, loaded_output, atol=1e-6)

        finally:
            # Clean up
            model_path.unlink()

    @pytest.mark.integration
    def test_inference_pipeline(self, test_images_dir: Path) -> None:
        """Test complete inference pipeline."""
        # Create and setup model
        model = UNet(in_ch=3, out_ch=3, base_ch=16)
        device = torch.device("cpu")
        model.eval()

        # Get test image
        test_images = list(test_images_dir.glob("*.png"))
        assert len(test_images) > 0

        input_image_path = test_images[0]

        # Run inference
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            output_path = Path(f.name)

        try:
            denoise_image(model, input_image_path, output_path, device)

            # Verify output
            assert output_path.exists()

            # Load and verify output image
            output_img = Image.open(output_path)
            input_img = Image.open(input_image_path)

            assert output_img.size == input_img.size
            assert output_img.mode == input_img.mode

        finally:
            # Clean up
            if output_path.exists():
                output_path.unlink()

    @pytest.mark.slow
    @pytest.mark.integration
    def test_training_step_simulation(self, test_images_dir: Path, sample_config: TrainConfig) -> None:
        """Test simulation of a training step."""
        # Create model and optimizer
        model = UNet(in_ch=3, out_ch=3, base_ch=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=sample_config.learning_rate)
        device = torch.device("cpu")

        # Create dataset
        dataset = PairdDataset(
            data_paths=test_images_dir,
            data_loading_fn=load_img,
            img_standardization_fn=standardize_img(),
            paring_fn=paring_clean_noisy(None, 0.1),
            limit=2,
        )

        # Create data loader
        from torch.utils.data import DataLoader

        dataloader = DataLoader(dataset, batch_size=sample_config.batch_size, shuffle=True)

        # Simulate training step
        model.train()

        for clean_batch, noisy_batch in dataloader:
            clean_batch = clean_batch.to(device)
            noisy_batch = noisy_batch.to(device)

            # Forward pass
            optimizer.zero_grad()
            losses = denoiser_loss(model, clean_batch, noisy_batch)
            loss = losses["Loss"]

            # Backward pass
            loss.backward()
            optimizer.step()

            # Verify loss is reasonable
            assert loss.item() >= 0
            assert not torch.isnan(loss)

            # Only test one batch
            break

    @pytest.mark.integration
    def test_config_integration_with_components(self, sample_config: TrainConfig) -> None:
        """Test configuration integration with various components."""
        # Test config values work with PyTorch
        learning_rate_tensor = torch.tensor(sample_config.learning_rate)
        assert learning_rate_tensor.item() == sample_config.learning_rate

        # Test batch size with tensor creation
        test_tensor = torch.rand(sample_config.batch_size, 3, sample_config.crop_size, sample_config.crop_size)
        assert test_tensor.shape[0] == sample_config.batch_size
        assert test_tensor.shape[2] == sample_config.crop_size
        assert test_tensor.shape[3] == sample_config.crop_size

        # Test paths exist and are Path objects
        assert isinstance(sample_config.output_dir, Path)
        assert isinstance(sample_config.log_dir, Path)

    @staticmethod
    @pytest.mark.integration
    def test_package_imports() -> None:
        """Test that all main components can be imported without errors."""
        # Test model imports
        from denoiser.models.unet import UNet

        assert UNet is not None

        # Test data imports
        from denoiser.data.data_loader import PairdDataset
        from denoiser.data.data_transformations import load_img, standardize_img

        assert PairdDataset is not None
        assert load_img is not None
        assert standardize_img is not None

        # Test inference imports
        from denoiser.inference import denoise_image, load_model

        assert denoise_image is not None
        assert load_model is not None

        # Test config imports
        from denoiser.configs.config import PairingKeyWords, TrainConfig

        assert TrainConfig is not None
        assert PairingKeyWords is not None

        # Test utils imports
        from denoiser.utils.calculate_loss import calculate_psnr, calculate_ssim, denoiser_loss

        assert calculate_psnr is not None
        assert calculate_ssim is not None
        assert denoiser_loss is not None
