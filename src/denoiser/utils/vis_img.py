from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from torch.utils.data import DataLoader

from denoiser.utils.alias import CanvasBufferEntry, IndexMapEntry


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


def _save_canvas(
    image_id: str,
    canvas: torch.Tensor,
    orig_h: int,
    orig_w: int,
    save_dir: Path,
    iteration: int,
    saved: int,
    destandardize_fn: Callable[[torch.Tensor | npt.NDArray[np.float32]], npt.NDArray[np.uint8]],
) -> int:
    stitched = canvas[:, :orig_h, :orig_w]
    stitched_u8 = destandardize_fn(stitched.detach().cpu())
    out_img = Image.fromarray(stitched_u8)
    safe_name = Path(image_id).stem
    out_img.save(save_dir / f"iter_{iteration:06d}_{safe_name}_pred.png")
    return saved + 1


def process_batch(
    batch: tuple[torch.Tensor, torch.Tensor, list[IndexMapEntry]],
    buffers: dict[str, CanvasBufferEntry],
    saved: int,
    max_images: int,
    model: torch.nn.Module,
    device: torch.device,
    save_dir: Path,
    iteration: int,
    destandardize_fn: Callable[[torch.Tensor | npt.NDArray[np.float32]], npt.NDArray[np.uint8]],
) -> int:
    _, noisy_tiles, metas = batch  # clean_tiles is unused

    noisy_tiles = torch.as_tensor(noisy_tiles).to(device)
    preds = model(noisy_tiles)  # (B, C, tile_h, tile_w)

    for i in range(preds.size(0)):
        mi = metas[i]
        image_id = str(mi["image_id"])

        # Initialize canvas for this image if needed
        if image_id not in buffers:
            buffers[image_id] = CanvasBufferEntry(
                canvas=torch.zeros(
                    (preds[i].shape[0], int(mi["padded_h"]), int(mi["padded_w"])),
                    dtype=preds[i].dtype,
                    device=preds[i].device,
                ),
                orig_h=int(mi["orig_h"]),
                orig_w=int(mi["orig_w"]),
                count=0,
                expected=int(mi["tiles_per_image"]) if "tiles_per_image" in mi else None,
            )

        canvas = buffers[image_id]["canvas"]
        assert isinstance(canvas, torch.Tensor)

        # Write prediction into canvas using inline meta access (reduces locals)
        canvas[
            :,
            int(mi["tile_y"]) : int(mi["tile_y"]) + int(mi["tile_h"]),
            int(mi["tile_x"]) : int(mi["tile_x"]) + int(mi["tile_w"]),
        ] = preds[i]
        buffers[image_id]["count"] = int(buffers[image_id]["count"] or 0) + 1

        count_val = buffers[image_id]["count"]
        expected_val = buffers[image_id]["expected"]
        # Ensure expected_val is not None and compare counts
        if expected_val is not None and int(count_val) >= int(expected_val):
            canvas = buffers[image_id]["canvas"]
            assert isinstance(canvas, torch.Tensor)
            orig_h_val = buffers[image_id]["orig_h"]
            orig_w_val = buffers[image_id]["orig_w"]

            saved = _save_canvas(
                image_id,
                canvas,
                int(orig_h_val),
                int(orig_w_val),
                save_dir,
                iteration,
                saved,
                destandardize_fn,
            )
            del buffers[image_id]
            if saved >= max_images:
                return saved
    return saved


def save_validation_predictions_stitched(
    model: torch.nn.Module,
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]]
    | DataLoader[
        tuple[
            npt.NDArray[np.uint8] | npt.NDArray[np.float32],
            npt.NDArray[np.uint8] | npt.NDArray[np.float32],
            IndexMapEntry,
        ]
    ],
    device: torch.device,
    destandardize_fn: Callable[[torch.Tensor | npt.NDArray[np.float32]], npt.NDArray[np.uint8]],
    save_dir: Path,
    iteration: int,
    max_images: int = 4,
) -> None:
    """Save validation prediction images by stitching tile predictions.

    When the val_loader returns tiles, save the predictions per original image
    (i.e., stitch the tile predictions and save one prediction image for each original image).
    """
    try:
        model.eval()
        save_dir.mkdir(parents=True, exist_ok=True)

        buffers: dict[str, CanvasBufferEntry] = {}
        saved = 0

        with torch.no_grad():
            for batch in val_loader:
                saved = process_batch(
                    batch, buffers, saved, max_images, model, device, save_dir, iteration, destandardize_fn
                )
                if saved >= max_images:
                    return

        # Save any remaining incomplete canvases
        for image_id, b in list(buffers.items()):
            if saved >= max_images:
                return
            canvas = b["canvas"]
            assert isinstance(canvas, torch.Tensor)
            orig_h = int(b["orig_h"])
            orig_w = int(b["orig_w"])
            saved = _save_canvas(
                image_id,
                canvas,
                orig_h,
                orig_w,
                save_dir,
                iteration,
                saved,
                destandardize_fn,
            )

    except (RuntimeError, OSError, ValueError, StopIteration, TypeError, KeyError) as e:
        print(f"Warning: Failed to save stitched validation predictions: {e}")
