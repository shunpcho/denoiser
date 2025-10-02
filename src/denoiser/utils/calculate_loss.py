import math

import torch


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


def denoiser_loss(
    model: torch.nn.Module,
    clean: torch.Tensor,
    noisy: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Calculate MSE loss for denoising task.

    Args:
        model: The denoising model
        clean: Clean ground truth images
        noisy: Noisy input images

    Returns:
        Dictionary containing MSE loss and metrics
    """
    outputs = model(noisy)
    mse_loss = torch.nn.functional.mse_loss(outputs, clean)

    # Calculate metrics for monitoring
    psnr = calculate_psnr(outputs, clean)
    ssim = calculate_ssim(outputs, clean)

    losses = {
        "Loss": mse_loss,  # Main loss is MSE
        "MSE": mse_loss,   # Same as Loss for clarity
        "PSNR": torch.tensor(psnr, device=mse_loss.device),
        "SSIM": torch.tensor(ssim, device=mse_loss.device),
    }

    return losses
