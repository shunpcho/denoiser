import math
from typing import Literal

import torch
import torch.nn.functional as f


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Calculate Peak Signal-to-Noise Ratio (PSNR) between two images."""
    mse = torch.nn.functional.mse_loss(img1, img2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(1.0 / math.sqrt(mse.item()))


def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> float:
    """Calculate Structural Similarity Index (SSIM) between two images."""
    # Simplified SSIM calculation
    mu1 = torch.nn.functional.avg_pool2d(img1, window_size, padding=window_size // 2)
    mu2 = torch.nn.functional.avg_pool2d(img2, window_size, padding=window_size // 2)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = torch.nn.functional.avg_pool2d(img1 * img1, window_size, padding=window_size // 2) - mu1_sq
    sigma2_sq = torch.nn.functional.avg_pool2d(img2 * img2, window_size, padding=window_size // 2) - mu2_sq
    sigma12 = torch.nn.functional.avg_pool2d(img1 * img2, window_size, padding=window_size // 2) - mu1_mu2

    c1 = 0.01**2
    c2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean().item()


def calculate_mse(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Calculate Mean Squared Error (MSE) between two images."""
    return torch.nn.functional.mse_loss(img1, img2).item()


class SSIMLoss(torch.nn.Module):
    """SSIM loss that matches calculate_ssim() implementation in calculate_loss.py.

    - window: avg_pool2d with kernel_size=window_size
    - c1 = 0.01**2, c2 = 0.03**2  (assumes images in [0, 1])
    - returns: 1 - mean(ssim_map)
    """

    def __init__(self, window_size: int = 11, eps: float = 1e-12, reduction: Literal["mean", "none"] = "mean") -> None:
        super().__init__()
        self.window_size = window_size
        self.eps = eps
        if reduction not in {"mean", "none"}:
            msg = "reduction must be 'mean' or 'none'"
            raise ValueError(msg)
        self.reduction = reduction

    def _compute_ssim_map(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Compute the SSIM map for two images."""
        ws = self.window_size
        pad = ws // 2

        mu1 = f.avg_pool2d(img1, ws, padding=pad)
        mu2 = f.avg_pool2d(img2, ws, padding=pad)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = f.avg_pool2d(img1 * img1, ws, padding=pad) - mu1_sq
        sigma2_sq = f.avg_pool2d(img2 * img2, ws, padding=pad) - mu2_sq
        sigma12 = f.avg_pool2d(img1 * img2, ws, padding=pad) - mu1_mu2

        c1 = 0.01**2
        c2 = 0.03**2

        num = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
        den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)

        return num / (den + self.eps)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        # img1/img2: (N, C, H, W), float, range [0,1] assumed
        ssim_map = self._compute_ssim_map(img1, img2)
        ssim = ssim_map.mean(dim=(-1, -2, -3))

        if self.reduction == "mean":
            ssim = ssim.mean()
            return 1.0 - ssim
        else:
            # Return (N,)
            return 1.0 - ssim
