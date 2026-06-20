"""Sanity tests for the toy-model engine.

These are fast, deterministic checks of the building blocks — not full training
runs. They guard the data statistics, the model shapes, and the geometric
metrics against accidental breakage.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

from superposition import ToyModel, generate_batch
from superposition import metrics


def test_sparsity_controls_fraction_of_zeros():
    g = torch.Generator().manual_seed(0)
    x = generate_batch(20_000, 10, sparsity=0.8, generator=g)
    frac_zero = (x == 0).float().mean().item()
    assert abs(frac_zero - 0.8) < 0.02


def test_dense_world_has_no_zeros():
    g = torch.Generator().manual_seed(0)
    x = generate_batch(1000, 10, sparsity=0.0, generator=g)
    assert (x > 0).all()


def test_model_shapes_and_relu():
    model = ToyModel(n_features=5, n_hidden=2)
    assert model.W.shape == (2, 5)
    x = torch.rand(8, 5)
    out = model(x)
    assert out.shape == (8, 5)
    assert (out >= 0).all()  # ReLU output is non-negative


def test_orthogonal_features_have_unit_dimensionality():
    # Two orthonormal columns in 2D -> each owns a full dimension.
    W = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # (m=2, n=2)
    dims = metrics.feature_dimensionality(W)
    assert torch.allclose(dims, torch.ones(2), atol=1e-5)
    assert metrics.frustration_energy(W) < 1e-6  # orthogonal -> no frustration


def test_regular_pentagon_dimensionality_is_two_fifths():
    angles = 2 * np.pi * np.arange(5) / 5
    W = torch.tensor(np.stack([np.cos(angles), np.sin(angles)]), dtype=torch.float32)
    dims = metrics.feature_dimensionality(W)
    assert torch.allclose(dims, torch.full((5,), 0.4), atol=1e-4)


def test_total_dimensionality_never_exceeds_hidden():
    torch.manual_seed(0)
    W = torch.randn(3, 20)  # 20 features crammed into 3 dimensions
    assert metrics.total_dimensionality(W) <= 3.0 + 1e-4
