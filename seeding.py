"""Centralized seeding utilities for reproducible multi-seed experiments.

A single run is parameterized by an integer ``seed``. We derive:
  * global RNG seeds for ``random``, ``numpy`` and ``torch`` (set once per run),
  * a per-episode SUMO seed so traffic varies across episodes but is reproducible
    across runs given the same ``seed``,
  * a fixed per-run evaluation seed so every evaluation within a run is measured on
    identical traffic (clean learning curves) while different runs see different
    evaluation traffic (captures variance for mean +/- std aggregation).
"""

import os
import random

import numpy as np


def set_global_seeds(seed: int) -> None:
    """Seed Python, NumPy and (if available) PyTorch RNGs for a run, and put PyTorch in
    a fully deterministic, single-threaded mode.

    The tabular methods (pure NumPy) are bit-reproducible from the seed alone. QMIX,
    however, was not: multi-threaded CPU reductions are order-nondeterministic and those
    tiny float differences compound through training into different policies. Forcing
    single-threaded execution and deterministic kernels makes QMIX bit-reproducible
    run-to-run. NumPy results are unaffected, so this does not change tabular outputs.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # required for CUDA matmul determinism
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.set_num_threads(1)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:  # older PyTorch without the warn_only kwarg
            torch.use_deterministic_algorithms(True)
    except ImportError:
        pass


def episode_seed(seed: int, episode: int) -> int:
    """Deterministic SUMO seed for a given training episode within a run."""
    return seed * 100_000 + episode


def eval_seed(seed: int) -> int:
    """Fixed SUMO seed used for every evaluation within a run."""
    return 10_000 + seed
