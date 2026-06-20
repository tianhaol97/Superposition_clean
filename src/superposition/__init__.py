"""A toy model of superposition, viewed through a statistical-physics lens.

See the README for the full story. The public API is intentionally small:

    from superposition import ToyModel, TrainConfig, train
    from superposition import metrics, viz
"""

from . import metrics, viz
from .data import generate_batch
from .model import ToyModel
from .train import TrainConfig, train, importance_weights

__all__ = [
    "ToyModel",
    "TrainConfig",
    "train",
    "importance_weights",
    "generate_batch",
    "metrics",
    "viz",
]
