"""TensorBoard logging utilities for denoiser training."""

from __future__ import annotations

from pathlib import Path
from typing import Self, TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import torch

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import TracebackType

    import numpy as np
    import numpy.typing as npt
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter

try:
    from torch.utils.tensorboard import SummaryWriter as _SummaryWriter
except ImportError:
    _SummaryWriter = None


class TensorBoard:
    """TensorBoard logging wrapper for training metrics and model visualization."""

    def __init__(
        self,
        log_dir: Path | str,
        dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        device: torch.device,
        crop_size: int | tuple[int, int],
        destandardize_img_fn: Callable[[npt.NDArray[np.float32] | torch.Tensor], npt.NDArray[np.uint8]],
        max_outputs: int = 4,
    ) -> None:
        """Initialize TensorBoard logger.

        Args:
            log_dir: Directory to save TensorBoard logs
            dataloader: DataLoader for getting sample images
            device: Device to run computations on
            crop_size: Size of image crops
            destandardize_img_fn: Function to convert normalized images back to uint8
            max_outputs: Maximum number of images to log
        """
        self.dataloader = dataloader
        self.device = device
        self.crop_size = crop_size if isinstance(crop_size, tuple) else (crop_size, crop_size)
        self.destandardize_img_fn = destandardize_img_fn
        self.max_outputs = max_outputs
        self.writer: SummaryWriter | None = None

        if _SummaryWriter is None:
            print("Warning: TensorBoard not available. Install with 'pip install tensorboard'")
            self.writer = None
        else:
            self.log_dir = Path(log_dir)
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = _SummaryWriter(log_dir=str(self.log_dir))

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value.

        Args:
            tag: Name of the scalar (e.g., 'Train/Loss', 'Val/Loss')
            value: Scalar value to log
            step: Global step value (typically iteration number)
        """
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: dict[str, float], step: int) -> None:
        """Log multiple related scalar values.

        Args:
            main_tag: Main tag for the group (e.g., 'Loss', 'Metrics')
            tag_scalar_dict: Dictionary of tag-value pairs
            step: Global step value
        """
        if self.writer is not None:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_histogram(self, tag: str, values: torch.Tensor, step: int) -> None:
        """Log histogram of tensor values.

        Args:
            tag: Name of the histogram
            values: Tensor containing values to histogram
            step: Global step value
        """
        if self.writer is not None:
            self.writer.add_histogram(tag, values, step)

    def log_image(self, tag: str, img_tensor: torch.Tensor, step: int, dataformats: str = "CHW") -> None:
        """Log an image.

        Args:
            tag: Name of the image
            img_tensor: Image tensor (C, H, W) format by default
            step: Global step value
            dataformats: Format of the image tensor ('CHW', 'HWC', etc.)
        """
        if self.writer is not None:
            self.writer.add_image(tag, img_tensor, step, dataformats=dataformats)

    @staticmethod
    def _process_tensor_for_display(tensor_batch: torch.Tensor) -> torch.Tensor:
        """Process tensor batch for TensorBoard display.

        Clamps values to [0, 1] range for proper visualization without destandardization.

        Args:
            tensor_batch: Batch of tensors (N, C, H, W)

        Returns:
            Processed tensor batch normalized to [0, 1] for TensorBoard
        """
        # Simply clamp the values to [0, 1] range for visualization
        # This avoids the complex destandardization process that's causing issues
        processed_batch = torch.clamp(tensor_batch, 0, 1)
        return processed_batch

    @staticmethod
    def _concatenate_images_horizontally(tensor_batch: torch.Tensor) -> torch.Tensor:
        """Concatenate batch images horizontally into a single image.

        Args:
            tensor_batch: Batch of images (N, C, H, W)

        Returns:
            Single image with all batch images concatenated horizontally (C, H, W*N)
        """
        if tensor_batch.size(0) == 1:
            return tensor_batch.squeeze(0)  # Remove batch dimension for single image

        # Concatenate along width dimension (dim=3 -> dim=2 after removing batch dim)
        concatenated = torch.cat([tensor_batch[i] for i in range(tensor_batch.size(0))], dim=2)
        return concatenated

    def log_images(self, model: torch.nn.Module, step: int, tag_prefix: str = "Images") -> None:
        """Log sample images using dataloader, model predictions, and destandardization.

        Images are arranged vertically in the order: Noisy, Clean, Predictions.
        Each row contains horizontally concatenated batch images.

        Args:
            model: The denoising model to generate predictions
            step: Global step value
            tag_prefix: Prefix for the image tags in TensorBoard
        """
        if self.writer is None:
            return

        model.eval()
        with torch.no_grad():
            try:
                # Get a batch of data from the dataloader
                batch = next(iter(self.dataloader))
                clean_images, noisy_images = batch

                # Move to device and limit to max_outputs
                clean_images = clean_images[: self.max_outputs].to(self.device)
                noisy_images = noisy_images[: self.max_outputs].to(self.device)

                # Generate predictions
                predictions = model(noisy_images)

                # Process tensors for display (clamp to [0, 1])
                clean_batch = TensorBoard._process_tensor_for_display(clean_images.cpu())
                noisy_batch = TensorBoard._process_tensor_for_display(noisy_images.cpu())
                pred_batch = TensorBoard._process_tensor_for_display(predictions.cpu())

                # Concatenate images horizontally for each type
                noisy_row = self._concatenate_images_horizontally(noisy_batch)
                clean_row = self._concatenate_images_horizontally(clean_batch)
                pred_row = self._concatenate_images_horizontally(pred_batch)

                # Concatenate vertically: Noisy -> Clean -> Predictions
                combined_image = torch.cat([noisy_row, clean_row, pred_row], dim=1)  # dim=1 is height

                # Log the combined image to TensorBoard
                self.writer.add_image(f"{tag_prefix}/Comparison", combined_image, step, dataformats="CHW")

            except (StopIteration, RuntimeError, ValueError) as e:
                print(f"Warning: Failed to log images to TensorBoard: {e}")

        model.train()

    def log_model_graph(self, model: torch.nn.Module, input_tensor: torch.Tensor) -> None:
        """Log model computational graph.

        Args:
            model: PyTorch model
            input_tensor: Sample input tensor for the model
        """
        if self.writer is not None:
            try:
                self.writer.add_graph(model, input_tensor)
            except (RuntimeError, ValueError) as e:
                print(f"Warning: Failed to add model graph to TensorBoard: {e}")

    def log_model_weights(self, model: torch.nn.Module, step: int) -> None:
        """Log model weights and gradients as histograms.

        Args:
            model: PyTorch model
            step: Global step value
        """
        if self.writer is not None:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # Log weights
                    self.writer.add_histogram(f"Weights/{name}", param.data, step)
                    # Log gradients if available
                    if param.grad is not None:
                        self.writer.add_histogram(f"Gradients/{name}", param.grad, step)

    def log_learning_rate(self, optimizer: torch.optim.Optimizer, step: int) -> None:
        """Log current learning rate.

        Args:
            optimizer: PyTorch optimizer
            step: Global step value
        """
        if self.writer is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                lr = param_group["lr"]
                self.writer.add_scalar(f"Learning_Rate/Group_{i}", lr, step)

    def log_training_metrics(
        self,
        train_loss: float,
        val_loss: float | None,
        learning_rate: float,
        step: int,
    ) -> None:
        """Log common training metrics.

        Args:
            train_loss: Training loss value
            val_loss: Validation loss value (optional)
            learning_rate: Current learning rate
            step: Global step value
        """
        if self.writer is not None:
            self.log_scalar("Train/Loss", train_loss, step)
            if val_loss is not None:
                self.log_scalar("Val/Loss", val_loss, step)
            self.log_scalar("Learning_Rate", learning_rate, step)

    def log_sample_predictions(
        self,
        clean_images: torch.Tensor,
        noisy_images: torch.Tensor,
        predictions: torch.Tensor,
        step: int,
        max_samples: int = 4,
    ) -> None:
        """Log sample image predictions for visual comparison.

        Args:
            clean_images: Ground truth clean images (N, C, H, W)
            noisy_images: Input noisy images (N, C, H, W)
            predictions: Model predictions (N, C, H, W)
            step: Global step value
            max_samples: Maximum number of samples to log
        """
        if self.writer is not None:
            # Limit number of samples to avoid overwhelming TensorBoard
            n_samples = min(max_samples, clean_images.size(0))

            # Ensure values are in [0, 1] range for proper visualization
            clean_viz = torch.clamp(clean_images[:n_samples], 0, 1)
            noisy_viz = torch.clamp(noisy_images[:n_samples], 0, 1)
            pred_viz = torch.clamp(predictions[:n_samples], 0, 1)

            self.writer.add_images("Sample/Clean", clean_viz, step)
            self.writer.add_images("Sample/Noisy", noisy_viz, step)
            self.writer.add_images("Sample/Predictions", pred_viz, step)

    def close(self) -> None:
        """Close the TensorBoard writer."""
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def __enter__(self) -> Self:
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Context manager exit."""
        self.close()
