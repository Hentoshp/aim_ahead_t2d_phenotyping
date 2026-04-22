"""SHAP importance computation for Module 2 (per-resample)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


@dataclass
class SHAPResult:
    shap_values: List[np.ndarray]
    summary: pd.DataFrame


def _normalize_shap_values(raw_shap_values, n_features: int) -> list[np.ndarray]:
    """Normalize SHAP outputs to one 2D array per class.

    Supported formats:
    - list of arrays: [(n_samples, n_features), ...]
    - ndarray: (n_samples, n_features)
    - ndarray: (n_samples, n_features, n_classes)
    - ndarray: (n_samples, n_classes, n_features)
    - Explanation-like object with `.values`
    """

    values = getattr(raw_shap_values, "values", raw_shap_values)

    if isinstance(values, list):
        normalized = [np.asarray(v) for v in values]
    else:
        arr = np.asarray(values)
        if arr.ndim == 2:
            normalized = [arr]
        elif arr.ndim == 3:
            if arr.shape[1] == n_features:
                normalized = [arr[:, :, class_idx] for class_idx in range(arr.shape[2])]
            elif arr.shape[2] == n_features:
                normalized = [arr[:, class_idx, :] for class_idx in range(arr.shape[1])]
            else:
                raise ValueError(
                    f"Unsupported 3D SHAP array shape {arr.shape}; expected one axis to match n_features={n_features}"
                )
        else:
            raise ValueError(f"Unsupported SHAP values ndim={arr.ndim}; expected 2D or 3D output.")

    for idx, arr in enumerate(normalized):
        if arr.ndim != 2:
            raise ValueError(f"Normalized SHAP array at class {idx} is not 2D: shape={arr.shape}")
        if arr.shape[1] != n_features:
            raise ValueError(
                f"Normalized SHAP array at class {idx} has {arr.shape[1]} features; expected {n_features}"
            )
    return normalized


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

    import shap

    explainer = shap.TreeExplainer(clf)
    shap_vals = _normalize_shap_values(explainer.shap_values(X), n_features=X.shape[1])

    summaries = []
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
