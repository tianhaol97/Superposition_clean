"""The toy model of superposition.

The model is deliberately the simplest thing that can exhibit superposition: a
linear "squeeze" into a low-dimensional bottleneck followed by a linear
"un-squeeze" and a ReLU.

    h    = W x            # compress n features into m < n dimensions
    x'   = ReLU(Wᵀ h + b) # reconstruct the original n features

There is *one* weight matrix ``W`` of shape ``(m, n)`` used in both directions
(a tied-weight autoencoder). Column ``i`` of ``W`` is the m-dimensional vector
the network uses to represent feature ``i``; we call it the feature's
**representation vector**.

Superposition is the situation where the network represents more features than
it has dimensions (more nonzero columns than ``m``) by packing those column
vectors into *overlapping*, non-orthogonal directions. The whole point of this
repo is to watch that packing happen and to characterise its geometry.
"""

from __future__ import annotations

import torch
from torch import nn


class ToyModel(nn.Module):
    """Tied-weight linear autoencoder with a ReLU output nonlinearity.

    Args:
        n_features: number of input features ``n``.
        n_hidden: bottleneck dimension ``m`` (``m < n`` is the interesting case).
    """

    def __init__(self, n_features: int, n_hidden: int):
        super().__init__()
        self.n_features = n_features
        self.n_hidden = n_hidden
        # W has shape (m, n): columns are the per-feature representation vectors.
        self.W = nn.Parameter(torch.empty(n_hidden, n_features))
        self.b = nn.Parameter(torch.zeros(n_features))
        nn.init.xavier_normal_(self.W)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Compress features into the m-dimensional bottleneck: h = W x."""
        return x @ self.W.T

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """Reconstruct features from the bottleneck: x' = ReLU(Wᵀ h + b)."""
        return torch.relu(h @ self.W + self.b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))
