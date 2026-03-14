#!/usr/bin/env python3
"""Test script to verify the collate function works correctly."""

import numpy as np
import torch

from denoiser.utils.data_utils import collate_fn


def test_collate_fn():
    """Test that the collate function properly converts numpy arrays to tensors."""
    # Create sample data similar to what the dataset returns
    batch = []
    for i in range(2):  # Simulate batch of 2
        # Create sample images as numpy arrays (C, H, W format)
        clean_img = np.random.rand(3, 64, 64).astype(np.float32)
        noisy_img = np.random.rand(3, 64, 64).astype(np.float32)
        batch.append((clean_img, noisy_img))

    # Test the collate function
    try:
        clean_batch, noisy_batch = collate_fn(batch)

        print("✓ Collate function executed successfully!")
        print(f"Clean batch shape: {clean_batch.shape}")
        print(f"Noisy batch shape: {noisy_batch.shape}")
        print(f"Clean batch type: {type(clean_batch)}")
        print(f"Noisy batch type: {type(noisy_batch)}")

        # Verify outputs are tensors
        assert isinstance(clean_batch, torch.Tensor), "Clean batch should be a tensor"
        assert isinstance(noisy_batch, torch.Tensor), "Noisy batch should be a tensor"

        # Verify batch dimensions
        assert clean_batch.shape[0] == 2, "Batch size should be 2"
        assert noisy_batch.shape[0] == 2, "Batch size should be 2"
        assert clean_batch.shape[1:] == (3, 64, 64), "Image shape should be (3, 64, 64)"
        assert noisy_batch.shape[1:] == (3, 64, 64), "Image shape should be (3, 64, 64)"

        print("✓ All tests passed!")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


if __name__ == "__main__":
    test_collate_fn()
