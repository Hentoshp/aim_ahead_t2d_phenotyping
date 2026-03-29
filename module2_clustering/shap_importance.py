"""SHAP importance scaffolding for Module 2."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class SHAPResult:
    shap_values: pd.DataFrame
    summary: pd.DataFrame


def compute_shap_distributions(
    X: np.ndarray,
    cluster_labels: np.ndarray,
    feature_names: list[str],
    random_state: int | None = None,
) -> SHAPResult:
    """Compute SHAP distributions per cluster. Placeholder only."""

    raise NotImplementedError("SHAP computation not implemented yet")

