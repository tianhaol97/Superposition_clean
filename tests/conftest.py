"""Test setup shared across all test modules.

Pytest imports this before collecting tests, so it is the right place to pin
threading. Single-threaded execution sidesteps a teardown deadlock that the
duplicate macOS OpenMP runtimes (NumPy's MKL + PyTorch) can trigger when the
process exits; the test workloads are tiny, so there is no speed cost.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch

torch.set_num_threads(1)
