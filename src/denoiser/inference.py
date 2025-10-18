from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image

from denoiser.data.data_transformations import (
    destandardize_img,
    load_img,
    standardize_img,
)
from denoiser.models.simple_unet import UNet

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy.typing as npt


def preprocess_image(
    image_path: Path,
    device: torch.device,
    standardize_fn: Callable[[npt.NDArray[np.uint8]], npt.NDArray[np.float32]],
) -> torch.Tensor:
    """Preprocess input image for inference.

    Args:
        image_path: Path to input image
        device: Device to load tensor on
        standardize_fn: Function to standardize image

    Returns:
        Preprocessed image tensor of shape (1, 3, H, W)
    """
    # Load image
    img = load_img(image_path)

    # Standardize (normalize to [0, 1] range)
    img_normalized = standardize_fn(img)

    # Convert to tensor and add batch dimension
    # img_normalized is (H, W, C), convert to (C, H, W) then add batch dim
    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)

    return img_tensor.to(device)


def postprocess_image(
    output_tensor: torch.Tensor,
    destandardize_fn: Callable[[torch.Tensor], npt.NDArray[np.uint8]],
) -> npt.NDArray[np.uint8]:
    """Postprocess model output to image.

    Args:
        output_tensor: Model output tensor of shape (1, 3, H, W)
        destandardize_fn: Function to destandardize image

    Returns:
        Processed image array of shape (H, W, 3)
    """
    # Remove batch dimension and move to CPU
    output = output_tensor.squeeze(0).cpu()

    # Destandardize (convert back to [0, 255] range)
    # Add batch dimension back for destandardize function
    output_batch = output.unsqueeze(0)
    img_denormalized = destandardize_fn(output_batch)

    # Convert from tensor to numpy array and remove batch dimension
    # Expected shape: (1, H, W, C) -> (H, W, C)
    # Note: destandardize_fn already returns numpy array, no need to call .numpy()
    ndim_batch = 4
    img_array = img_denormalized.squeeze(0) if len(img_denormalized.shape) == ndim_batch else img_denormalized

    return img_array.astype(np.uint8)


def load_model(
    model_path: Path,
    device: torch.device,
    in_ch: int = 3,
    out_ch: int = 3,
    base_ch: int = 64,
) -> UNet:
    """Load trained denoiser model.

    Args:
        model_path: Path to model checkpoint
        device: Device to load model on
        in_ch: Number of input channels
        out_ch: Number of output channels
        base_ch: Base number of channels

    Returns:
        Loaded model in evaluation mode

    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    if not model_path.exists():
        msg = f"Model file not found: {model_path}"
        raise FileNotFoundError(msg)

    # Initialize model
    model = UNet(in_ch=in_ch, out_ch=out_ch, base_ch=base_ch)
    # TODO: Make model architecture configurable

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Move to device and set to evaluation mode
    model = model.to(device)
    model.eval()

    return model


def denoise_image(
    model: UNet,
    image_path: Path,
    output_path: Path,
    device: torch.device,
) -> None:
    """Denoise a single image using trained model.

    Args:
        model: Trained denoiser model
        image_path: Path to input noisy image
        output_path: Path to save denoised image
        device: Device to run inference on
    """
    # Create preprocessing and postprocessing functions
    # Using same normalization as training (mean=0, std=1)
    standardize_fn = standardize_img(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))
    destandardize_fn = destandardize_img(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))

    # Preprocess image
    input_tensor = preprocess_image(image_path, device, standardize_fn)

    # Run inference
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # Postprocess output
    denoised_img = postprocess_image(output_tensor, destandardize_fn)

    # Save result
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(denoised_img).save(output_path)

    print(f"Denoised image saved to: {output_path}")


def batch_denoise(
    model: UNet,
    input_dir: Path,
    output_dir: Path,
    device: torch.device,
    image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif"),
) -> None:
    """Denoise all images in a directory.

    Args:
        model: Trained denoiser model
        input_dir: Directory containing noisy images
        output_dir: Directory to save denoised images
        device: Device to run inference on
        image_extensions: Supported image file extensions

    Raises:
        FileNotFoundError: If input directory does not exist
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        msg = f"Input directory not found: {input_dir}"
        raise FileNotFoundError(msg)

    # Find all image files
    image_files: list[Path] = []
    for ext in image_extensions:
        image_files.extend(input_dir.glob(f"*{ext}"))
        image_files.extend(input_dir.glob(f"*{ext.upper()}"))

    if not image_files:
        print(f"No image files found in {input_dir}")
        return

    print(f"Found {len(image_files)} images to process")

    # Process each image
    for i, image_path in enumerate(image_files, 1):
        output_path = output_dir / f"denoised_{image_path.name}"

        try:
            denoise_image(model, image_path, output_path, device)
            print(f"Processed {i}/{len(image_files)}: {image_path.name}")
        except (FileNotFoundError, OSError, RuntimeError) as e:
            print(f"Error processing {image_path.name}: {e}")


if __name__ == "__main__":
    # Configuration
    parser = argparse.ArgumentParser(description="Denoise images using trained model")
    parser.add_argument("--model", type=Path, required=True, help="Path to model checkpoint")
    parser.add_argument("--input", type=Path, required=True, help="Input image or directory")
    parser.add_argument("--output", type=Path, required=True, help="Output image or directory")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (cpu/cuda/auto)")
    args = parser.parse_args()
    model_path = args.model
    input_image = args.input
    output_image = args.output

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        print("Please train a model first or provide the correct path.")
        error_msg = f"Model file not found: {model_path}"
        raise FileNotFoundError(error_msg)

    model = load_model(model_path, device)
    print(f"Model loaded successfully from {model_path}")

    # Denoise image
    if not input_image.exists():
        print(f"Input image not found: {input_image}")
        print("Please provide a valid image path.")
        error_msg = f"Input image not found: {input_image}"
        raise FileNotFoundError(error_msg)

    batch_denoise(model, input_image, output_image, device)
    print("Denoising completed successfully!")
