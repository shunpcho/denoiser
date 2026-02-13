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

            def _ier_loss(
                pred: torch.Tensor,
                target: torch.Tensor,
                sfl: SFLLoss = self.sfl,
                mse: nn.Module = self.mse,
            ) -> torch.Tensor:
                return sfl(pred, target) + mse(pred, target)

            self.loss_fn = _ier_loss
        else:
            msg = f"Unsupported loss type: {loss_type}"
            raise ValueError(msg)

    def compute_components(self, predicted: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.loss_type == "ier":
            assert self.sfl is not None
            assert self.mse is not None
            sfl_loss = self.sfl(predicted, target)
            mse_loss = self.mse(predicted, target)
            loss = sfl_loss + mse_loss
            esfl = self.sfl.e_sfl(predicted, target).mean(dim=0)
        else:
            loss = self.loss_fn(predicted, target)
            esfl = self.esfl_monitor.e_sfl(predicted, target).mean(dim=0)

        components: dict[str, torch.Tensor] = {"Loss": loss}
        for idx, value in enumerate(esfl):
            components[f"ESFL_{idx}"] = value
        return components

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.compute_components(predicted, target)["Loss"]
