"""Shared test fixtures and configuration."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from denoiser.configs.config import PairingKeyWords, TrainConfig


@pytest.fixture
def temp_image_dir() -> Path:
    """Create temporary directory with test images.

    Yields:
        Path: Directory containing test images.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create test images
        for i in range(3):
            img = Image.new("RGB", (64, 64), color=(i * 80, i * 80, i * 80))
            img.save(tmp_path / f"test_image_{i}.png")

        yield tmp_path


@pytest.fixture
def sample_config() -> TrainConfig:
    """Create a sample training configuration."""
    return TrainConfig(
        batch_size=4,
        crop_size=32,
        learning_rate=1e-4,
        loss_type="l2",
        iteration=100,
        interval=10,
    )


@pytest.fixture
def sample_pairing_keywords() -> PairingKeyWords:
    """Create sample pairing keywords."""
    return PairingKeyWords(clean="clean", noisy="noisy")


@pytest.fixture
def sample_image() -> np.ndarray:
    """Create a sample test image array.

    Returns:
        np.ndarray: Sample RGB image as uint8 array.
    """
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)


@pytest.fixture
def sample_tensor() -> torch.Tensor:
    """Create a sample test tensor.

    Returns:
        torch.Tensor: Sample tensor with shape (1, 3, 32, 32).
    """
    torch.manual_seed(42)
    return torch.rand(1, 3, 32, 32)


@pytest.fixture
def device() -> torch.device:
    """Get appropriate device for testing.

    Returns:
        torch.device: CPU or CUDA device depending on availability.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest settings."""
    # Add custom markers
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "integration: marks integration tests")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark GPU tests
        if "gpu" in item.nodeid.lower() or "cuda" in item.nodeid.lower():
            item.add_marker(pytest.mark.gpu)

        # Mark slow tests
        if "slow" in item.nodeid.lower() or any(
            keyword in item.name.lower() for keyword in ["batch", "integration", "full_pipeline"]
        ):
            item.add_marker(pytest.mark.slow)
