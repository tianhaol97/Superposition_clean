"""Synthetic sparse feature data.

The toy model lives in a world of ``n`` independent "features". On any given
example, most features are *off* (exactly zero) and only a few are *on*. How
often a feature is off is controlled by the ``sparsity`` knob:

    sparsity = P(a given feature is zero on a given example)

so ``sparsity = 0`` means every feature is always on (a dense world) and
``sparsity -> 1`` means features almost never co-occur (a very sparse world).

Sparsity is the single most important control parameter in this project: it is
the knob that drives the network between qualitatively different "phases" of
how it stores information, much like temperature drives phase transitions in a
physical system.
"""

from __future__ import annotations

import torch


def generate_batch(
    batch_size: int,
    n_features: int,
    sparsity: float,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Draw a batch of sparse feature vectors.

    Each feature is independently set to zero with probability ``sparsity``.
    When a feature is *on*, its value is drawn uniformly from ``[0, 1)``.

    Args:
        batch_size: number of examples.
        n_features: number of features ``n`` per example.
        sparsity: probability that any individual feature is zero, in ``[0, 1)``.
        generator: optional torch RNG for reproducibility.
        device: device to place the tensor on.

    Returns:
        Tensor of shape ``(batch_size, n_features)`` with values in ``[0, 1)``,
        most of which are exactly zero when ``sparsity`` is large.
    """
    if not 0.0 <= sparsity < 1.0:
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")

    values = torch.rand(
        batch_size, n_features, generator=generator, device=device
    )
    mask = (
        torch.rand(batch_size, n_features, generator=generator, device=device)
        >= sparsity
    )
    return values * mask
