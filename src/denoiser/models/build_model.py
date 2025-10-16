from pathlib import Path

import torch
from torch import nn

from denoiser.models.unet import UNet


def create_model(
    model_name: str | None, in_channels: int = 3, out_channels: int | None = None, pretrained: bool = True
) -> nn.Module:
    """Create UNet model with standard parameters."""
    if model_name == "vit_base_patch16_224":
        return UNet(
            encoder_name=model_name,
            encoder_depth=4,
            in_channels=in_channels,
            out_channels=out_channels,
            pretrained=pretrained,
        )
    return UNet(encoder_name="resnet34", in_channels=in_channels, out_channels=out_channels, pretrained=pretrained)


def load_model_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model
