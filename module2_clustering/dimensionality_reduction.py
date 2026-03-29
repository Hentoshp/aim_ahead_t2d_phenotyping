"""PCA wrapper for Module 2: fit to variance threshold and persist artifacts."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import joblib


@dataclass
class PCAResult:
    transformed: np.ndarray
    pca_model: any  # sklearn.decomposition.PCA at implementation time
    explained_variance: float


def run_pca(
    matrix: pd.DataFrame,
    variance_threshold: float,
    random_state: int | None = None,
) -> PCAResult:
    """Fit PCA to reach variance_threshold and return transformed data."""

    if not 0 < variance_threshold <= 1:
        raise ValueError("variance_threshold must be in (0, 1].")

    if not isinstance(matrix, pd.DataFrame):
        raise TypeError("matrix must be a pandas DataFrame.")

    if matrix.isnull().any().any():
        raise ValueError("clustering_matrix contains nulls; upstream QC should prevent this.")

    non_numeric = [c for c in matrix.columns if not pd.api.types.is_numeric_dtype(matrix[c])]
    if non_numeric:
        raise ValueError(f"All features must be numeric; non-numeric columns found: {non_numeric}")

    # Determine number of components to hit the variance threshold.
    pca_full = PCA(random_state=random_state, svd_solver="full")
    pca_full.fit(matrix.values)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumvar, variance_threshold) + 1)
    n_components = min(n_components, matrix.shape[1])

    pca = PCA(n_components=n_components, random_state=random_state, svd_solver="full")
    transformed = pca.fit_transform(matrix.values)
    explained = float(pca.explained_variance_ratio_.sum())

    return PCAResult(transformed=transformed, pca_model=pca, explained_variance=explained)


def save_pca_artifacts(
    result: PCAResult,
    artifacts_path: Path,
    feature_names: list[str] | None = None,
) -> Path:
    """Persist PCA model and summary metadata. Returns path to saved model."""

    artifacts_path.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_path / "pca_model.joblib"
    summary_path = artifacts_path / "pca_summary.json"

    joblib.dump(result.pca_model, model_path)

    rs = getattr(result.pca_model, "random_state", None)
    try:
        rs_serializable = int(rs) if rs is not None else None
    except Exception:
        rs_serializable = None

    summary = {
        "n_components": getattr(result.pca_model, "n_components_", None),
        "n_features": getattr(result.pca_model, "n_features_in_", None),
        "explained_variance": result.explained_variance,
        "variance_ratio": result.pca_model.explained_variance_ratio_.tolist(),
        "singular_values": result.pca_model.singular_values_.tolist(),
        "random_state": rs_serializable,
        "svd_solver": result.pca_model.svd_solver,
        "feature_names": feature_names or [],
        "created": datetime.now(timezone.utc).isoformat(),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return model_path
