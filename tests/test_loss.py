"""Tests for loss calculation utilities."""

from __future__ import annotations

import pytest
import torch

from denoiser.utils.calculate_loss import (
    calculate_mse,
    calculate_psnr,
    calculate_ssim,
    denoiser_loss,
)


class TestLossFunctions:
    """Test loss calculation functions."""

    @pytest.fixture
    def sample_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Create sample tensors for testing."""
        torch.manual_seed(42)
        img1 = torch.rand(1, 3, 32, 32)
        img2 = torch.rand(1, 3, 32, 32)
        return img1, img2

    @pytest.fixture
    def identical_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Create identical tensors for testing."""
        torch.manual_seed(42)
        img = torch.rand(1, 3, 32, 32)
        return img, img.clone()

    def test_calculate_mse_different_images(self, sample_tensors: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Test MSE calculation with different images."""
        img1, img2 = sample_tensors
        mse = calculate_mse(img1, img2)

        assert isinstance(mse, float)
        assert mse > 0  # Different images should have non-zero MSE

    def test_calculate_mse_identical_images(self, identical_tensors: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Test MSE calculation with identical images."""
        img1, img2 = identical_tensors
        mse = calculate_mse(img1, img2)

        assert isinstance(mse, float)
        assert mse == pytest.approx(0.0, abs=1e-6)  # Identical images should have zero MSE

    def test_calculate_psnr_different_images(self, sample_tensors: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Test PSNR calculation with different images."""
        img1, img2 = sample_tensors
        psnr = calculate_psnr(img1, img2)

        assert isinstance(psnr, float)
        assert psnr > 0  # PSNR should be positive

    def test_calculate_psnr_identical_images(self, identical_tensors: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Test PSNR calculation with identical images."""
        img1, img2 = identical_tensors
        psnr = calculate_psnr(img1, img2)

        assert isinstance(psnr, float)
        # Identical images should have very high PSNR (approaching infinity)
        assert psnr > 100  # Should be much higher than typical PSNR values

    def test_calculate_ssim_different_images(self, sample_tensors: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Test SSIM calculation with different images."""
        img1, img2 = sample_tensors
        ssim = calculate_ssim(img1, img2)

        assert isinstance(ssim, float)
        assert 0 <= ssim <= 1  # SSIM should be between 0 and 1

    def test_calculate_ssim_identical_images(self, identical_tensors: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Test SSIM calculation with identical images."""
        img1, img2 = identical_tensors
        ssim = calculate_ssim(img1, img2)

        assert isinstance(ssim, float)
        assert ssim == pytest.approx(1.0, abs=1e-3)  # Identical images should have SSIM = 1

    @pytest.mark.parametrize("window_size", [7, 11, 15])
    def test_calculate_ssim_different_window_sizes(
        self, sample_tensors: tuple[torch.Tensor, torch.Tensor], window_size: int
    ) -> None:
        """Test SSIM with different window sizes."""
        img1, img2 = sample_tensors
        ssim = calculate_ssim(img1, img2, window_size=window_size)

        assert isinstance(ssim, float)
        assert 0 <= ssim <= 1

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_loss_functions_different_batch_sizes(self, batch_size: int) -> None:
        """Test loss functions with different batch sizes."""
        torch.manual_seed(42)
        img1 = torch.rand(batch_size, 3, 32, 32)
        img2 = torch.rand(batch_size, 3, 32, 32)

        mse = calculate_mse(img1, img2)
        psnr = calculate_psnr(img1, img2)
        ssim = calculate_ssim(img1, img2)

        assert isinstance(mse, float)
        assert isinstance(psnr, float)
        assert isinstance(ssim, float)
        assert mse > 0
        assert psnr > 0
        assert 0 <= ssim <= 1

    def test_loss_functions_gpu_compatibility(self, sample_tensors: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Test loss functions work on GPU if available."""
        img1, img2 = sample_tensors

        if torch.cuda.is_available():
            img1_cuda = img1.cuda()
            img2_cuda = img2.cuda()

            mse = calculate_mse(img1_cuda, img2_cuda)
            psnr = calculate_psnr(img1_cuda, img2_cuda)
            ssim = calculate_ssim(img1_cuda, img2_cuda)

            assert isinstance(mse, float)
            assert isinstance(psnr, float)
            assert isinstance(ssim, float)

    def test_loss_functions_different_dtypes(self) -> None:
        """Test loss functions with different tensor dtypes."""
        torch.manual_seed(42)
        img1_float32 = torch.rand(1, 3, 32, 32, dtype=torch.float32)
        img2_float32 = torch.rand(1, 3, 32, 32, dtype=torch.float32)

        mse = calculate_mse(img1_float32, img2_float32)
        psnr = calculate_psnr(img1_float32, img2_float32)
        ssim = calculate_ssim(img1_float32, img2_float32)

        assert isinstance(mse, float)
        assert isinstance(psnr, float)
        assert isinstance(ssim, float)


class TestDenoiserLoss:
    """Test denoiser_loss function."""

    @pytest.fixture
    def mock_model(self) -> torch.nn.Module:
        """Create a simple mock denoising model."""

        class MockDenoiser(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, kernel_size=3, padding=1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.conv(x)

        return MockDenoiser()

    def test_denoiser_loss_computation(self, mock_model: torch.nn.Module) -> None:
        """Test denoiser loss computation."""
        torch.manual_seed(42)
        clean = torch.rand(2, 3, 32, 32)
        noisy = torch.rand(2, 3, 32, 32)

        losses = denoiser_loss(mock_model, clean, noisy)

        assert isinstance(losses, dict)
        assert "Loss" in losses
        assert "MSE" in losses
        assert "PSNR" in losses
        assert "SSIM" in losses

        # Check that all values are tensors
        assert isinstance(losses["Loss"], torch.Tensor)
        assert isinstance(losses["MSE"], torch.Tensor)
        assert isinstance(losses["PSNR"], torch.Tensor)
        assert isinstance(losses["SSIM"], torch.Tensor)

        # Loss and MSE should be the same
        assert torch.allclose(losses["Loss"], losses["MSE"])

    def test_denoiser_loss_gradient_computation(self, mock_model: torch.nn.Module) -> None:
        """Test that denoiser loss allows gradient computation."""
        torch.manual_seed(42)
        clean = torch.rand(2, 3, 32, 32)
        noisy = torch.rand(2, 3, 32, 32)

        losses = denoiser_loss(mock_model, clean, noisy)
        loss = losses["Loss"]

        # Backward pass should work
        loss.backward()

        # Check that model parameters have gradients
        for param in mock_model.parameters():
            assert param.grad is not None

    def test_denoiser_loss_device_consistency(self, mock_model: torch.nn.Module) -> None:
        """Test that loss tensors are on the same device as inputs."""
        torch.manual_seed(42)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        mock_model = mock_model.to(device)
        clean = torch.rand(2, 3, 32, 32).to(device)
        noisy = torch.rand(2, 3, 32, 32).to(device)

        losses = denoiser_loss(mock_model, clean, noisy)

        # All loss tensors should be on the same device
        for loss_tensor in losses.values():
            assert loss_tensor.device == device

    def test_denoiser_loss_shapes(self, mock_model: torch.nn.Module) -> None:
        """Test denoiser loss with different input shapes."""
        torch.manual_seed(42)

        for batch_size in [1, 2, 4]:
            for height, width in [(32, 32), (64, 64), (16, 16)]:
                clean = torch.rand(batch_size, 3, height, width)
                noisy = torch.rand(batch_size, 3, height, width)

                losses = denoiser_loss(mock_model, clean, noisy)

                # Should return valid losses for any reasonable input shape
                assert "Loss" in losses
                assert losses["Loss"].item() >= 0

    def test_denoiser_loss_identical_inputs(self, mock_model: torch.nn.Module) -> None:
        """Test denoiser loss when model output equals ground truth."""

        # Create a model that returns input unchanged (identity)
        class IdentityModel(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x

        identity_model = IdentityModel()
        torch.manual_seed(42)
        clean = torch.rand(2, 3, 32, 32)

        losses = denoiser_loss(identity_model, clean, clean)  # Same input as target

        # MSE should be zero when output equals target
        assert losses["MSE"].item() == pytest.approx(0.0, abs=1e-6)
        assert losses["SSIM"].item() == pytest.approx(1.0, abs=1e-3)

    @staticmethod
    def test_denoiser_loss_values_reasonable() -> None:
        """Test that loss values are in reasonable ranges."""

        # Simple mock model
        class SimpleModel(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Add small noise
                return x + 0.1 * torch.randn_like(x)

        model = SimpleModel()
        torch.manual_seed(42)
        clean = torch.rand(2, 3, 32, 32)
        noisy = torch.rand(2, 3, 32, 32)

        losses = denoiser_loss(model, clean, noisy)

        # Loss values should be reasonable (not NaN or infinite)
        assert torch.isfinite(losses["Loss"])
        assert torch.isfinite(losses["PSNR"])
        assert torch.isfinite(losses["SSIM"])
        assert losses["Loss"].item() >= 0
        assert losses["PSNR"].item() >= 0
        assert 0 <= losses["SSIM"].item() <= 1
