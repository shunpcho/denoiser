import timm
import torch
from torch import nn

ENCODER_DEPTHS = {3, 4, 5}


class ConvBlock(nn.Module):
    """Double convolution block used in decoder."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """UNet with TIMM encoder backbone.

    Args:
        encoder_name: Name of the TIMM model (e.g., 'resnet34', 'efficientnet_b0')
        encoder_depth: Number of encoder stages to use (3, 4, or 5)
        in_channels: Number of input channels (default: 3)
        decoder_channels: List of channel numbers for decoder blocks
        pretrained: Whether to use pretrained weights for the encoder
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        in_channels: int = 3,
        encoder_depth: int = 5,
        out_channels: int | None = None,
        decoder_channels: list[int] | None = None,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        if encoder_depth not in ENCODER_DEPTHS:
            msg = f"Encoder depth must be one of {ENCODER_DEPTHS}, got {encoder_depth}."
            raise ValueError(msg)
        self.encoder_depth = encoder_depth
        self.in_channels = in_channels

        # Set output channels
        if out_channels is None:
            out_channels = in_channels
        self.out_channels = out_channels

        # Set decoder channels if not provided
        if decoder_channels is None:
            decoder_channels = [512, 256, 128, 64, 32]

        # Create encoder from TIMM
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=in_channels,
            out_indices=list(range(encoder_depth)),
        )

        # Get encoder output channels for each stage
        encoder_channels = self.encoder.feature_info.channels[:encoder_depth]

        # Adjust decoder channels to match encoder depth
        if len(decoder_channels) > encoder_depth:
            decoder_channels = decoder_channels[:encoder_depth]
        elif len(decoder_channels) < encoder_depth:
            decoder_channels += [decoder_channels[-1]] * (encoder_depth - len(decoder_channels))

        self.decoder_channels = decoder_channels

        # Build decoder
        self.decoder_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        # Reverse encoder channels for bottom-up decoding
        encoder_channels = encoder_channels[::-1]

        for i in range(len(decoder_channels)):
            in_ch = encoder_channels[0] if i == 0 else decoder_channels[i - 1]

            # Skip connection channel
            skip_ch = encoder_channels[i + 1] if i + 1 < len(encoder_channels) else 0

            # Total input channels for this block
            total_in_ch = in_ch + skip_ch

            self.upsamples.append(nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2))
            self.decoder_blocks.append(ConvBlock(total_in_ch, decoder_channels[i]))

        # Final segmentation head
        self.segmentation_head = nn.Conv2d(decoder_channels[-1], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the UNet model.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, out_channels, H, W)
        """
        input_size = x.shape[2:]

        # Encoder
        encoder_features = self.encoder(x)

        # Reverse for bottom-up decoding
        encoder_features = encoder_features[::-1]

        # Start from the deepest encoder feature
        d = encoder_features[0]

        # decoder with skip connections
        for i, (upsample, decoder_block) in enumerate(zip(self.upsamples, self.decoder_blocks, strict=False)):
            d = upsample(d)

            # Add skip connection if available
            if i + 1 < len(encoder_features):
                skip = encoder_features[i + 1]
                # Handle size mismatch
                if d.shape[2:] != skip.shape[2:]:
                    d = nn.functional.interpolate(d, size=skip.shape[2:], mode="bilinear", align_corners=False)
                d = torch.cat([d, skip], dim=1)

            d = decoder_block(d)

        # Final segmentation
        output = self.segmentation_head(d)

        # Upsample to input size if needed
        if output.shape[2:] != input_size:
            print(f"Upsampling output from {output.shape} to {input_size}")
            output = nn.functional.interpolate(output, size=input_size, mode="bilinear", align_corners=False)

        return output


if __name__ == "__main__":
    print("=== Same Input/Output Channels (Image-to-Image) ===")
    # RGB to RGB (image restoration, enhancement, etc.)
    model = UNet(
        encoder_name="resnet34",
        in_channels=3,
        out_channels=3,  # Same as input
        decoder_channels=[256, 128, 64, 32, 16],
    )
    x = torch.randn(2, 3, 256, 256)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Channels match: {output.shape[1] == x.shape[1]}")

    # print("\n=== Auto-match channels (default behavior) ===")
    # # If out_channels is None, it automatically matches in_channels
    # model = UNet(
    #     encoder_name="resnet34",
    #     in_channels=4,  # e.g., RGBA
    #     # out_channels not specified, will default to 4
    #     decoder_channels=[256, 128, 64, 32, 16],
    # )
    # x = torch.randn(2, 4, 256, 256)
    # output = model(x)
    # print(f"Input: {x.shape[1]} channels, Output: {output.shape[1]} channels")

    # print("\n=== Single Channel (Grayscale) ===")
    # model = UNet(encoder_name="resnet34", in_channels=1, out_channels=1, decoder_channels=[256, 128, 64, 32, 16])
    # x = torch.randn(2, 1, 256, 256)
    # output = model(x)
    # print(f"Input shape: {x.shape}")
    # print(f"Output shape: {output.shape}")

    # print("\n=== Different Use Cases ===")

    # # Image denoising
    # denoising_model = UNet(in_channels=3, out_channels=3)
    # print("Denoising model: 3 → 3 channels")

    # # Depth estimation
    # depth_model = UNet(in_channels=3, out_channels=1)
    # print("Depth model: 3 → 1 channel")

    # # Super-resolution (channels stay same)
    # sr_model = UNet(in_channels=3, out_channels=3)
    # print("Super-resolution model: 3 → 3 channels")

    # # Segmentation
    # seg_model = UNet(in_channels=3, out_channels=10)
    # print("Segmentation model: 3 → 10 classes")

    # print("\n=== Model Parameters ===")
    # encoders = ["resnet34", "resnet50", "efficientnet_b0"]
    # for enc in encoders:
    #     model = UNet(encoder_name=enc, in_channels=3, out_channels=3, encoder_depth=5)
    #     params = sum(p.numel() for p in model.parameters()) / 1e6
    #     print(f"{enc}: {params:.2f}M parameters")
