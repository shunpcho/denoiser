import sys
from pathlib import Path

# Ensure package source is importable when tests run from workspace root
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "denoiser" / "src"))

import numpy as np
import torch

from denoiser.utils.data_utils import collate_fn


def test_collate_without_meta():
    a = np.zeros((3, 4, 4), dtype=np.float32)
    b = np.ones((3, 4, 4), dtype=np.float32)
    batch = [(a, b), (a, b)]
    clean_batch, noisy_batch = collate_fn(batch)
    assert isinstance(clean_batch, torch.Tensor)
    assert isinstance(noisy_batch, torch.Tensor)
    assert clean_batch.shape[0] == 2
    assert clean_batch.shape[1:] == (3, 4, 4)


def test_collate_with_meta():
    a = np.zeros((3, 4, 4), dtype=np.float32)
    b = np.ones((3, 4, 4), dtype=np.float32)
    meta1 = {"path": "p1"}
    batch = [(a, b, meta1)]
    clean_batch, noisy_batch, metas = collate_fn(batch)
    assert isinstance(metas, list)
    assert metas[0]["path"] == "p1"
