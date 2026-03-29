"""Gaussian Mixture Model utilities (scaffolding)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np


@dataclass
class GMMFit:
    model: any  # sklearn.mixture.GaussianMixture once implemented
    k: int
    covariance_type: str
    bic: float
    aic: float
    log_likelihood: float


def grid_search_gmm(
    X: np.ndarray,
    k_range: Iterable[int],
    covariance_types: Iterable[str],
    random_state: int | None = None,
) -> GMMFit:
    """Run grid search over K and covariance type. Placeholder only."""

    raise NotImplementedError("GMM grid search not implemented yet")


def predict_membership(fit: GMMFit, X: np.ndarray) -> np.ndarray:
    """Return soft membership probabilities for samples. Placeholder only."""

    raise NotImplementedError("GMM predict_membership not implemented yet")

