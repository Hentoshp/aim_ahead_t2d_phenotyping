"""Cluster profiling scaffolding for Module 2."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class ProfilingResult:
    membership: pd.DataFrame
    profiles: pd.DataFrame
    summary: dict


def build_membership_matrix(
    participant_ids: Iterable,
    membership_probs: np.ndarray,
    artifacts_path: Path | None = None,
) -> pd.DataFrame:
    """Return participant_id-indexed membership matrix."""

    participant_ids = list(participant_ids)
    if len(participant_ids) != membership_probs.shape[0]:
        raise ValueError("participant_ids length does not match membership rows")
    if not np.isfinite(membership_probs).all():
        raise ValueError("membership_probs contains non-finite values")

    k = membership_probs.shape[1]
    cols = [f"pi_{i+1}" for i in range(k)]
    membership_df = pd.DataFrame(membership_probs, index=participant_ids, columns=cols)

    # normalize tiny drift
    membership_df = membership_df.div(membership_df.sum(axis=1), axis=0)

    if artifacts_path:
        artifacts_path.mkdir(parents=True, exist_ok=True)
        membership_df.to_parquet(artifacts_path / "membership_matrix.parquet")

    return membership_df


def back_project_centroids(
    pca_model,
    gmm_model,
    feature_columns: list[str],
    artifacts_path: Path | None = None,
) -> pd.DataFrame:
    """Back-project GMM means into original feature space."""

    if not hasattr(gmm_model, "means_"):
        raise ValueError("gmm_model missing means_ attribute")
    if not hasattr(pca_model, "inverse_transform"):
        raise ValueError("pca_model missing inverse_transform")

    centroids_original = pca_model.inverse_transform(gmm_model.means_)
    k = gmm_model.means_.shape[0]

    records = []
    for cluster_idx in range(k):
        for feat, val in zip(feature_columns, centroids_original[cluster_idx]):
            records.append({"cluster": cluster_idx, "feature": feat, "centroid_value": float(val)})

    profiles_df = pd.DataFrame(records)

    if artifacts_path:
        artifacts_path.mkdir(parents=True, exist_ok=True)
        profiles_df.to_parquet(artifacts_path / "cluster_profiles.parquet")

    return profiles_df


def summarize_profiles(membership_df: pd.DataFrame, profiles_df: pd.DataFrame) -> dict:
    """Small JSON-friendly summary with cluster sizes and top features."""
    # soft sizes
    cluster_cols = [c for c in membership_df.columns if c.startswith("pi_")]
    soft_sizes = membership_df[cluster_cols].sum().to_dict()
    soft_sizes = {k: float(v) for k, v in soft_sizes.items()}

    # hard sizes
    hard_labels = membership_df[cluster_cols].idxmax(axis=1)
    hard_counts = hard_labels.value_counts().to_dict()
    hard_counts = {k: int(v) for k, v in hard_counts.items()}

    # top features per cluster by absolute centroid value
    top_feats = {}
    for c, dfc in profiles_df.groupby("cluster"):
        dfc_sorted = dfc.reindex(dfc["centroid_value"].abs().sort_values(ascending=False).index)
        top_feats[int(c)] = dfc_sorted.head(5)[["feature", "centroid_value"]].to_dict(orient="records")

    return {
        "cluster_soft_sizes": soft_sizes,
        "cluster_hard_sizes": hard_counts,
        "top_features_by_abs_centroid": top_feats,
    }
