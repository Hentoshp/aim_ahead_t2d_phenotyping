"""Cluster profiling scaffolding for Module 2."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class ProfilingResult:
    membership: pd.DataFrame
    profiles: pd.DataFrame


def build_membership_matrix(
    participant_ids: Iterable,
    membership_probs: np.ndarray,
) -> pd.DataFrame:
    """Return participant_id-indexed membership matrix. Placeholder only."""

    raise NotImplementedError("Membership matrix construction not implemented yet")


def back_project_centroids(
    pca_model,
    gmm_model,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Back-project GMM means into original feature space. Placeholder only."""

    raise NotImplementedError("Centroid back-projection not implemented yet")

