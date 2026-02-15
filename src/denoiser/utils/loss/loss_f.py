from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from image_analysis.subband_loss.IER import SFLLoss
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Callable


class LossFunction(nn.Module):
    def __init__(self, loss_type: str = "mse") -> None:
        super().__init__()
        self.loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

        self.loss_type = loss_type
        self.esfl_monitor = SFLLoss()
        self.sfl: SFLLoss | None = None
        self.mse: nn.Module | None = None
        if loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_type == "mae":
            self.loss_fn = nn.L1Loss()
        elif loss_type == "ier":
            self.sfl = SFLLoss()
            self.mse = nn.MSELoss()
            self.loss_fn = nn.MSELoss()
        else:
            msg = f"Unsupported loss type: {loss_type}"
            raise ValueError(msg)

    def _compute_ier_loss_and_esfl(
        self, predicted: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute IER loss and ESFL in a single pass of subband decomposition."""
        assert self.sfl is not None
        assert self.mse is not None

        e_sfl = self.sfl.e_sfl(predicted, target)
        e_l2 = torch.nn.functional.mse_loss(predicted, target, reduction="none").mean(dim=(1, 2, 3))
        w_sfl = (e_l2.unsqueeze(1) / e_sfl).mean(dim=0)
        sfl_loss = (w_sfl * e_sfl).mean()
        mse_loss = self.mse(predicted, target)

        loss = sfl_loss + mse_loss
        esfl_mean = e_sfl.mean(dim=0)
        return loss, esfl_mean

    def compute_components(self, predicted: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.loss_type == "ier":
            loss, esfl = self._compute_ier_loss_and_esfl(predicted, target)
        else:
            loss = self.loss_fn(predicted, target)
            with torch.no_grad():
                esfl = self.esfl_monitor.e_sfl(predicted, target).mean(dim=0)

        components: dict[str, torch.Tensor] = {"Loss": loss}
        for idx, value in enumerate(esfl):
            components[f"ESFL_{idx}"] = value.detach()
        return components

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "ier":
            return self._compute_ier_loss_and_esfl(predicted, target)[0]
        return self.loss_fn(predicted, target)
