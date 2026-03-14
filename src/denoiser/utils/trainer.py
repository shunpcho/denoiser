import shutil
import time
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader

from denoiser.configs.config import TrainConfig
from denoiser.utils.alias import BATCH_ENTRY_LENGTH_WITH_META, IndexMapEntry
from denoiser.utils.calculate_loss import calculate_mse, calculate_psnr, calculate_ssim
from denoiser.utils.loss.loss_f import LossFunction


class Trainer:
    """Trainer class for managing the training process of a model."""

    def __init__(self, device: torch.device) -> None:
        """Initialize the Trainer instance.

        Args:
            device: The device to run the training on (e.g., CPU or GPU).
        """
        self.device = device

    def loop(self, train: bool = True) -> dict[str, float]:
        msg = "Subclasses should implement the loop method to perform training or validation steps."
        raise NotImplementedError(msg)

    def train_step(self) -> dict[str, float]:
        """Perform a training batch."""
        return self.loop()

    def val_step(self) -> dict[str, float]:
        """Perform a validation batch and return loss, PSNR, and SSIM."""
        with torch.no_grad():
            return self.loop(train=False)

    @staticmethod
    def _print(
        step: int,
        max_iter: int,
        loss: torch.Tensor,
        step_time: float,
        fetch_time: float,
    ) -> None:
        """Print information related to the current step.

        Args:
            step: Current step (within the epoch).
            max_iter: Maximum number of iterations in the epoch.
            loss: Loss value for the current step.
            step_time: Time it took to perform the whole step.
            fetch_time: Time it took to load the data for the step.
        """
        pre_str = f"{step} / {max_iter} ["
        loss_str = f"] Loss: {loss.item():.3e}"
        timing_str = f"  -  Step time: {step_time:.2f}ms  -  Fetch time: {fetch_time:.2f}ms    "

        term_cols = shutil.get_terminal_size(fallback=(156, 38)).columns
        progress_bar_len = max(8, min(term_cols - len(pre_str) - len(loss_str) - len(timing_str) - 1, 30))
        progress = int(progress_bar_len * (step / max_iter))
        progress_bar_str = f"{progress * '='}>{(progress_bar_len - progress) * '.'}"

        full_string = pre_str + progress_bar_str + loss_str + timing_str
        print(full_string, end=("\r" if step < max_iter else "\n"), flush=True)


class TrainTrainer(Trainer):
    """Trainer class that handles training and validation epochs for the training task."""

    def __init__(
        self,
        models: torch.nn.Module,
        optimizers: torch.optim.Optimizer,
        train_config: TrainConfig,
        train_dataloader: DataLoader[
            tuple[npt.NDArray[np.uint8] | npt.NDArray[np.float32], npt.NDArray[np.uint8] | npt.NDArray[np.float32]]
        ],
        val_dataloader: DataLoader[
            tuple[
                npt.NDArray[np.uint8] | npt.NDArray[np.float32],
                npt.NDArray[np.uint8] | npt.NDArray[np.float32],
                IndexMapEntry,
            ]
        ],
    ) -> None:
        super().__init__(device=train_config.device)
        self.models = models
        self.optimizer = optimizers
        self.loss_type = train_config.loss_type
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = train_config.device
        self.use_amp = self.device.type == "cuda"
        self.grad_scaler = GradScaler(enabled=self.use_amp)

    def save_model(self, path: Path | str) -> None:
        """Save the model state dict to the specified path."""
        torch.save(
            {
                "model_state_dict": self.models.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def loop(self, train: bool = True) -> dict[str, float]:
        """Do pass on every step of the train or validation dataset.

        Args:
            train: Whether it is a train or validation loop

        Returns:
            Dictionary containing average losses for all loss components.
        """
        self.models.train(train)
        dataloader = self.train_dataloader if train else self.val_dataloader
        steps = len(dataloader)
        step_losses: dict[str, float] = {}
        loss_fn = LossFunction(loss_type=self.loss_type).to(self.device)

        step_time: float | None = None
        fetch_time: float | None = None
        step_start_time = time.perf_counter()

        for step, batch in enumerate(dataloader, 1):
            clean, noisy = self._extract_clean_noisy(batch)
            clean = clean.to(self.device, non_blocking=True)
            noisy = noisy.to(self.device, non_blocking=True)
            data_loading_finished_time = time.perf_counter()

            if train:
                self.optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=self.use_amp):
                outputs = self.models(noisy)
                loss_components = loss_fn.compute_components(outputs, clean)
            loss = loss_components["Loss"]

            if train:
                self._backward_and_step(loss)

            self._accumulate_losses(step_losses, loss_components)
            self._accumulate_metrics(step_losses, outputs, clean)

            previous_step_start_time = step_start_time
            current_time = time.perf_counter()
            if step_time is not None and fetch_time is not None:
                step_time = 0.9 * step_time + 0.1 * 1000 * (current_time - step_start_time)
                fetch_time = 0.9 * fetch_time + 0.1 * 1000 * (data_loading_finished_time - previous_step_start_time)
            else:
                step_time = 1000 * (current_time - step_start_time)
                fetch_time = 1000 * (data_loading_finished_time - previous_step_start_time)
            step_start_time = current_time

            self._print(
                step,
                steps,
                loss,
                step_time,
                fetch_time,
            )

        return {key: value / steps for key, value in step_losses.items()}

    @staticmethod
    def _extract_clean_noisy(
        batch: tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], IndexMapEntry],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # batch can be (clean, noisy) or (clean, noisy, meta)
        if len(batch) == BATCH_ENTRY_LENGTH_WITH_META:
            clean, noisy, _ = batch
        else:
            clean, noisy = batch[:2]
        clean = torch.as_tensor(clean)
        noisy = torch.as_tensor(noisy)
        return clean, noisy

    def _backward_and_step(self, loss: torch.Tensor) -> None:
        if self.use_amp:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

    @staticmethod
    def _accumulate_losses(step_losses: dict[str, float], loss_components: dict[str, torch.Tensor]) -> None:
        for name, value in loss_components.items():
            step_losses[name] = step_losses.get(name, 0.0) + value.item()

    @staticmethod
    def _accumulate_metrics(step_losses: dict[str, float], outputs: torch.Tensor, clean: torch.Tensor) -> None:
        with torch.no_grad():
            mse = calculate_mse(outputs, clean)
            psnr = calculate_psnr(outputs, clean)
            ssim = calculate_ssim(outputs, clean)
        step_losses["MSE"] = step_losses.get("MSE", 0.0) + mse
        step_losses["PSNR"] = step_losses.get("PSNR", 0.0) + psnr
        step_losses["SSIM"] = step_losses.get("SSIM", 0.0) + ssim

    def train_step(self) -> dict[str, float]:
        """Perform a training step."""
        return self.loop(train=True)

    def val_step(self) -> dict[str, float]:
        """Perform a validation step."""
        with torch.no_grad():
            step_loss = self.loop(train=False)
        return step_loss
