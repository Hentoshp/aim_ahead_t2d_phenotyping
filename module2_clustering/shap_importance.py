"""SHAP importance computation for Module 2 (per-resample)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import shap

@dataclass
class SHAPResult:
    shap_values: List[np.ndarray]
    summary: pd.DataFrame


def compute_shap_distributions(
    X: np.ndarray,
    cluster_labels: np.ndarray,
    feature_names: list[str],
    random_state: int | None = None,
) -> SHAPResult:
    """Compute SHAP distributions per cluster (single run).

    Returns:
      shap_values: list per class (as from shap TreeExplainer)
      summary: dataframe with columns [cluster, feature, shap_mean, abs_shap_mean]
    """

    if X is None or len(X) == 0:
        raise ValueError("X is empty")
    if cluster_labels is None or len(cluster_labels) != len(X):
        raise ValueError("cluster_labels length mismatch")
    if not np.isfinite(X).all():
        raise ValueError("X contains non-finite values")

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=random_state,
    )
    clf.fit(X, cluster_labels)

    explainer = shap.TreeExplainer(clf)
    shap_vals = explainer.shap_values(X)

    # shap returns list per class for multiclass
    summaries = []
    n_classes = len(shap_vals) if isinstance(shap_vals, list) else shap_vals.shape[1]
    if not isinstance(shap_vals, list):
        shap_vals = [shap_vals]  # binary with margin output

    for c, sv in enumerate(shap_vals):
        mean = sv.mean(axis=0)
        abs_mean = np.abs(sv).mean(axis=0)
        summaries.append(
            pd.DataFrame(
                {
                    "cluster": c,
                    "feature": feature_names,
                    "shap_mean": mean,
                    "abs_shap_mean": abs_mean,
                }
            )
        )

    summary_df = pd.concat(summaries, ignore_index=True)
    return SHAPResult(shap_values=shap_vals, summary=summary_df)
