"""Shared helpers for the experiment scripts."""

from __future__ import annotations

import os

# Work around a macOS clash where NumPy (MKL) and PyTorch each link their own
# OpenMP runtime. Pinning to a single thread sidesteps a teardown deadlock the
# duplicate runtimes can trigger at process exit; these models are tiny enough
# that single-threaded is just as fast. Must be set before torch/numpy import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")  # headless: save figures, never open a window

import torch  # noqa: E402

torch.set_num_threads(1)

from superposition import TrainConfig, train  # noqa: E402
from superposition.metrics import feature_dimensionality  # noqa: E402

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def figure_path(name: str) -> str:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    return os.path.normpath(os.path.join(FIGURES_DIR, name))


def train_best_of(config: TrainConfig, n_seeds: int = 5):
    """Train ``n_seeds`` independent runs and return the lowest-loss model.

    The loss landscape is non-convex and occasionally a run gets stuck in a
    poor local minimum; taking the best of a handful of seeds gives the clean,
    reproducible geometry we want to display.
    """
    best_model = None
    best_loss = float("inf")
    for s in range(n_seeds):
        cfg = replace(config, seed=config.seed + s, history=[])
        model = train(cfg)
        loss = cfg.history[-1]
        if loss < best_loss:
            best_loss, best_model = loss, model
    return best_model, best_loss


def log(msg: str) -> None:
    print(msg, flush=True, file=sys.stderr)
