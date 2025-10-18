from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from torch.utils.data import DataLoader


def save_validation_predictions(
    model: torch.nn.Module,
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    destandardize_fn: Callable[[torch.Tensor | npt.NDArray[np.float32]], npt.NDArray[np.uint8]],
    save_dir: Path,
    iteration: int,
    max_samples: int = 4,
) -> None:
    """Save validation prediction images.

    Args:
        model: The denoising model
        val_loader: Validation data loader
        device: Device to run inference on
        destandardize_fn: Function to convert normalized images back to uint8
        save_dir: Directory to save prediction images
        iteration: Current training iteration
        max_samples: Maximum number of samples to save
    """
    try:
        model.eval()
        save_dir.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            batch = next(iter(val_loader))
            clean_images, noisy_images = batch

            # Move to device and limit samples
            clean_images = clean_images[:max_samples].to(device)
            noisy_images = noisy_images[:max_samples].to(device)

            # Generate predictions
            predictions = model(noisy_images)

            # Save images
            for i in range(clean_images.size(0)):
                try:
                    # Convert tensors to numpy arrays using destandardize function
                    pred_np = destandardize_fn(predictions[i].cpu())

                    # Handle different image formats
                    grayscale_dims = 2
                    if clean_images[i].ndim == grayscale_dims:  # Grayscale
                        pred_img = Image.fromarray(pred_np, mode="L")
                    else:  # RGB
                        pred_img = Image.fromarray(pred_np)

                    # Save with descriptive filenames
                    pred_img.save(save_dir / f"iter_{iteration:06d}_sample_{i:02d}_pred.png")

                except (OSError, ValueError, RuntimeError) as e:
                    print(f"Warning: Failed to save sample {i}: {e}")
                    continue

    except (StopIteration, RuntimeError, OSError) as e:
        print(f"Warning: Failed to save validation predictions: {e}")
