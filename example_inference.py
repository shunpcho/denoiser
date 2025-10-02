#!/usr/bin/env python3
"""Example script demonstrating how to use the denoiser inference module."""

from pathlib import Path

import torch

from denoiser.inference import denoise_image, load_model


def main() -> None:
    """Example usage of denoiser inference."""
    # Configuration
    model_path = Path("./results/best_model.pth")
    input_image = Path("./data/noisy_image.jpg")  # Replace with your noisy image
    output_image = Path("./output/denoised_image.jpg")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    try:
        model = load_model(model_path, device)
        print(f"Model loaded successfully from {model_path}")
    except FileNotFoundError:
        print(f"Model file not found: {model_path}")
        print("Please train a model first or provide the correct path.")
        return
    
    # Denoise image
    try:
        denoise_image(model, input_image, output_image, device)
        print("Denoising completed successfully!")
    except FileNotFoundError:
        print(f"Input image not found: {input_image}")
        print("Please provide a valid image path.")
    except (OSError, RuntimeError, torch.cuda.OutOfMemoryError) as e:
        print(f"Error during inference: {e}")


if __name__ == "__main__":
    main()