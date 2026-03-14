"""Tests for denoiser models."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from denoiser.models.unet import ConvBlock, UNet


class TestConvBlock:
    """Test ConvBlock component."""

    def test_conv_block_init(self) -> None:
        """Test ConvBlock initialization."""
        block = ConvBlock(in_ch=3, out_ch=64)
        assert isinstance(block, nn.Module)
        assert hasattr(block, "conv1")
        assert hasattr(block, "conv2")
        assert hasattr(block, "relu")

    def test_conv_block_forward(self) -> None:
        """Test ConvBlock forward pass."""
        block = ConvBlock(in_ch=3, out_ch=64)
        x = torch.randn(1, 3, 64, 64)
        output = block(x)
        assert output.shape == (1, 64, 64, 64)

    @pytest.mark.parametrize(("in_ch", "out_ch"), [(1, 32), (3, 64), (64, 128)])
    def test_conv_block_different_channels(self, in_ch: int, out_ch: int) -> None:
        """Test ConvBlock with different channel configurations."""
        block = ConvBlock(in_ch=in_ch, out_ch=out_ch)
        x = torch.randn(2, in_ch, 32, 32)
        output = block(x)
        assert output.shape == (2, out_ch, 32, 32)


class TestUNet:
    """Test UNet model."""

    def test_unet_init(self) -> None:
        """Test UNet initialization."""
        model = UNet(in_ch=3, out_ch=3, base_ch=64)
        assert isinstance(model, nn.Module)

        # Check encoder layers
        assert hasattr(model, "enc1")
        assert hasattr(model, "enc2")
        assert hasattr(model, "enc3")
        assert hasattr(model, "enc4")

        # Check decoder layers
        assert hasattr(model, "up3")
        assert hasattr(model, "up2")
        assert hasattr(model, "up1")

        # Check other components
        assert hasattr(model, "pool")
        assert hasattr(model, "final_conv")

    def test_unet_forward(self) -> None:
        """Test UNet forward pass."""
        model = UNet(in_ch=3, out_ch=3, base_ch=64)
        x = torch.randn(1, 3, 256, 256)
        output = model(x)

        # Output should have same spatial dimensions as input
        assert output.shape == (1, 3, 256, 256)

    @pytest.mark.parametrize(("batch_size", "height", "width"), [(1, 128, 128), (2, 256, 256), (4, 64, 64)])
    def test_unet_different_input_sizes(self, batch_size: int, height: int, width: int) -> None:
        """Test UNet with different input sizes."""
        model = UNet(in_ch=3, out_ch=3, base_ch=32)  # Smaller base_ch for faster testing
        x = torch.randn(batch_size, 3, height, width)
        output = model(x)

        assert output.shape == (batch_size, 3, height, width)

    @pytest.mark.parametrize(("in_ch", "out_ch", "base_ch"), [(1, 1, 32), (3, 3, 64), (3, 1, 32)])
    def test_unet_different_channel_configs(self, in_ch: int, out_ch: int, base_ch: int) -> None:
        """Test UNet with different channel configurations."""
        model = UNet(in_ch=in_ch, out_ch=out_ch, base_ch=base_ch)
        x = torch.randn(1, in_ch, 128, 128)
        output = model(x)

        assert output.shape == (1, out_ch, 128, 128)

    def test_unet_gradient_flow(self) -> None:
        """Test that gradients can flow through UNet."""
        model = UNet(in_ch=3, out_ch=3, base_ch=32)
        x = torch.randn(1, 3, 128, 128, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Check that input has gradients
        assert x.grad is not None
        assert x.grad.shape == x.shape

        # Check that model parameters have gradients
        for param in model.parameters():
            assert param.grad is not None

    @staticmethod
    def test_unet_eval_mode() -> None:
        """Test UNet in evaluation mode."""
        model = UNet(in_ch=3, out_ch=3, base_ch=32)
        model.eval()

        x = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 3, 128, 128)

    @staticmethod
    def test_unet_device_compatibility() -> None:
        """Test UNet device compatibility."""
        model = UNet(in_ch=3, out_ch=3, base_ch=32)

        # Test CPU
        x_cpu = torch.randn(1, 3, 64, 64)
        output_cpu = model(x_cpu)
        assert output_cpu.device.type == "cpu"

        # Test CUDA if available
        if torch.cuda.is_available():
            model_cuda = model.cuda()
            x_cuda = torch.randn(1, 3, 64, 64).cuda()
            output_cuda = model_cuda(x_cuda)
            assert output_cuda.device.type == "cuda"

    @staticmethod
    def test_unet_parameter_count() -> None:
        """Test UNet parameter count is reasonable."""
        model = UNet(in_ch=3, out_ch=3, base_ch=64)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Should have a reasonable number of parameters
        assert total_params > 1000  # At least 1K parameters
        assert total_params < 50_000_000  # Less than 50M parameters
        assert total_params == trainable_params  # All parameters should be trainable by default
