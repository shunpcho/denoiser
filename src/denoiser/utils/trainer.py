import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional
from torch.utils.data import DataLoader

from denoiser.configs.config import TrainConfig


class Trainer:
    """Trainer class for managing the training process of a model."""

    def __init__(self, device: torch.device) -> None:
        """Initialize the Trainer instance.

        Args:
            device: The device to run the training on (e.g., CPU or GPU).
        """
        self.device = device

    def loop(self, *, train: bool = True) -> float: ...

    def train_step(self) -> float:
        """Perform a training batch."""
        return self.loop

    def val_step(self) -> tuple[float, float, float]:
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
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
    ) -> None:
        super().__init__(device=train_config.device)
        self.models = models
        self.optimizer = optimizers
        self.train_configs = train_config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = train_config.device

    def save_model(self, path: Path | str) -> None:
        """Save the model state dict to the specified path."""
        torch.save(
            {
                "model_state_dict": self.models.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def loop(self, *, train: bool = True) -> dict[str, float]:
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

        step_time: float | None = None
        fetch_time: float | None = None
        step_start_time = time.perf_counter()

        for step, (clean, noisy) in enumerate(dataloader, 1):
            clean = clean.to(self.device, non_blocking=True)
            noisy = noisy.to(self.device, non_blocking=True)
            data_loading_finished_time = time.perf_counter()

            if train:
                self.optimizer.zero_grad()

            outputs = self.models(noisy)
            loss = torch.nn.functional.mse_loss(outputs, clean)

            if train:
                loss.backward()
                self.optimizer.step()

            step_losses["Loss"] = step_losses.get("Loss", 0) + loss.item()

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

    def train_step(self) -> dict[str, float]:
        """Perform a training step."""
        return self.loop(train=True)

    def val_step(self) -> dict[str, float]:
        """Perform a validation step."""
        with torch.no_grad():
            step_loss = self.loop(train=False)
        return step_loss
