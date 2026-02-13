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
        if loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_type == "mae":
            self.loss_fn = nn.L1Loss()
        elif loss_type == "ier":
            sfl = SFLLoss()
            mse = nn.MSELoss()
            self.loss_fn = lambda pred, target: sfl(pred, target) + mse(pred, target)
        else:
            msg = f"Unsupported loss type: {loss_type}"
            raise ValueError(msg)

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(predicted, target)
