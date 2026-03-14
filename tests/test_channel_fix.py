#!/usr/bin/env python3

import sys

sys.path.append("/home/s.chochi/denoiser/src")

import numpy as np
import torch

from denoiser.utils.data_utils import collate_fn


def test_channel_ordering():
    """Test that the collate function correctly handles HWC to CHW conversion."""
    print("Testing channel ordering fix...")

    # Create sample data in HWC format (Height, Width, Channels)
    height, width, channels = 512, 512, 3

    # Create mock batch data in HWC format (like what comes from image loading)
    batch = []
    for i in range(2):  # batch size 2
        clean_img = np.random.rand(height, width, channels).astype(np.float32)
        noisy_img = np.random.rand(height, width, channels).astype(np.float32)
        batch.append((clean_img, noisy_img))

    print(f"Original image shape (HWC): {batch[0][0].shape}")

    # Apply collate function
    clean_batch, noisy_batch = collate_fn(batch)

    print(f"Batched tensor shape (NCHW): {clean_batch.shape}")
    print("Expected shape: torch.Size([2, 3, 512, 512])")

    # Verify the shape is correct for PyTorch models
    expected_shape = torch.Size([2, 3, 512, 512])

    if clean_batch.shape == expected_shape and noisy_batch.shape == expected_shape:
        print("✓ Channel ordering fix successful!")
        print(f"Clean batch shape: {clean_batch.shape}")
        print(f"Noisy batch shape: {noisy_batch.shape}")
        print("✓ Ready for UNet input (NCHW format)")
        return True
    else:
        print("✗ Channel ordering fix failed!")
        print(f"Got: {clean_batch.shape}")
        print(f"Expected: {expected_shape}")
        return False


if __name__ == "__main__":
    success = test_channel_ordering()
    if success:
        print("\n✓ All channel ordering tests passed!")
    else:
        print("\n✗ Channel ordering tests failed!")
        sys.exit(1)
