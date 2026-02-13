from pathlib import Path

import torch
from torch import nn
from vit_unet.config.model_config import VitunetConfig
from vit_unet.models.model import ViTUNet

from denoiser.models.unet import UNet


def create_model(
    model_name: str | None, in_channels: int = 3, out_channels: int | None = None, pretrained: bool = True
) -> nn.Module:
    """Create UNet model with standard parameters."""
    if model_name == "vit_unet":
        config = VitunetConfig(
            depth=2,
            depth_te=2,
            size_bottleneck=2,
            preprocessing="conv",
            im_size=256,
            patch_size=32,
            num_channels=3,
            hidden_dim=128,
            num_heads=8,
            attn_drop=0.2,
            proj_drop=0.2,
            linear_drop=0,
            verbose=False,
        )
        return ViTUNet(config)
    return UNet(encoder_name="resnet34", in_channels=in_channels, out_channels=out_channels, pretrained=pretrained)


def load_model_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model
