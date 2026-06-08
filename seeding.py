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
    """Seed Python, NumPy and (if available) PyTorch RNGs for a run."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def episode_seed(seed: int, episode: int) -> int:
    """Deterministic SUMO seed for a given training episode within a run."""
    return seed * 100_000 + episode


def eval_seed(seed: int) -> int:
    """Fixed SUMO seed used for every evaluation within a run."""
    return 10_000 + seed
