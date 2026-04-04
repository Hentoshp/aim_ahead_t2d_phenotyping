"""PCA wrapper for Module 2: fit to variance threshold and persist artifacts."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.decomposition import PCA


def correlation_filter(df: pd.DataFrame, threshold: float = 0.9, method: str = "pearson") -> tuple[pd.DataFrame, list[str]]:
    """Drop columns with |r| > threshold, keeping the first occurrence."""
    if threshold <= 0 or threshold >= 1:
        raise ValueError("threshold must be in (0, 1)")
    if df.empty:
        return df.copy(), []

    corr = df.corr(method=method).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop: list[str] = [col for col in upper.columns if any(upper[col] > threshold)]
    filtered = df.drop(columns=to_drop)
    return filtered, to_drop


@dataclass
class PCAResult:
    transformed: pd.DataFrame
    pca_model: PCA  # sklearn.decomposition.PCA
    explained_variance: float
    n_components: int


def run_pca(
    matrix: pd.DataFrame,
    variance_threshold: float | None,
    random_state: int | None = None,
    artifacts_path: Path | None = None,
    mode: str = "variance",
    n_components: int | None = None,
) -> PCAResult:
    """Fit PCA either to a variance threshold or fixed n_components."""

    if not isinstance(matrix, pd.DataFrame):
        raise TypeError("matrix must be a pandas DataFrame.")

    if matrix.isnull().any().any():
        raise ValueError("clustering_matrix contains nulls; upstream QC should prevent this.")

    non_numeric = [c for c in matrix.columns if not pd.api.types.is_numeric_dtype(matrix[c])]
    if non_numeric:
        raise ValueError(f"All features must be numeric; non-numeric columns found: {non_numeric}")

    if not np.isfinite(matrix.values).all():
        raise ValueError("clustering_matrix contains non-finite values.")

    if mode not in ("variance", "fixed"):
        raise ValueError("mode must be 'variance' or 'fixed'.")

    if mode == "variance":
        if variance_threshold is None or not 0 < variance_threshold <= 1:
            raise ValueError("variance_threshold must be in (0, 1] when mode=='variance'.")

        pca_full = PCA(random_state=random_state, svd_solver="full")
        pca_full.fit(matrix.values)
        cumvar = np.cumsum(pca_full.explained_variance_ratio_)
        n_components_calc = int(np.searchsorted(cumvar, variance_threshold) + 1)
    else:
        if n_components is None or n_components < 1:
            raise ValueError("n_components must be provided and >=1 when mode=='fixed'.")
        n_components_calc = int(n_components)

    n_components_final = max(1, min(n_components_calc, matrix.shape[1]))

    pca = PCA(n_components=n_components_final, random_state=random_state, svd_solver="full")
    transformed_arr = pca.fit_transform(matrix.values)
    explained = float(pca.explained_variance_ratio_.sum())

    # Optional cumulative variance plot (post-fit to avoid duplicated work)
    if artifacts_path is not None:
        artifacts_path.mkdir(parents=True, exist_ok=True)
        try:
            import matplotlib.pyplot as plt  # local import to avoid hard dep during tests

            cumvar_plot = np.cumsum(pca.explained_variance_ratio_)
            plt.figure(figsize=(6, 4))
            plt.plot(np.arange(1, len(cumvar_plot) + 1), cumvar_plot, marker="o", linewidth=1.2)
            if variance_threshold is not None:
                plt.axhline(variance_threshold, color="red", linestyle="--", linewidth=1, label=f"threshold={variance_threshold:.2f}")
            plt.xlabel("Number of components")
            plt.ylabel("Cumulative explained variance")
            plt.ylim(0, 1.01)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(Path(artifacts_path) / "pca_cumulative_variance.png", dpi=150)
            plt.close()
        except ImportError:
            pass

    pc_names = [f"PC{i+1}" for i in range(transformed_arr.shape[1])]
    transformed_df = pd.DataFrame(transformed_arr, index=matrix.index, columns=pc_names)

    return PCAResult(
        transformed=transformed_df,
        pca_model=pca,
        explained_variance=explained,
        n_components=n_components_final,
    )


def save_pca_artifacts(
    result: PCAResult,
    artifacts_path: Path,
    feature_names: list[str] | None = None,
    save_transformed: bool = True,
    mode: str = "variance",
    variance_threshold: float | None = None,
    n_components_requested: int | None = None,
) -> Path:
    """Persist PCA model and summary metadata. Returns path to saved model."""

    artifacts_path.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_path / "pca_model.joblib"
    summary_path = artifacts_path / "pca_summary.json"
    transformed_path = artifacts_path / "pca_transformed.parquet"
    loadings_path = artifacts_path / "pca_loadings.parquet"

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
        "mode": mode,
        "variance_threshold": variance_threshold,
        "n_components_requested": n_components_requested,
        "created": datetime.now(timezone.utc).isoformat(),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    if save_transformed:
        result.transformed.to_parquet(transformed_path)

    # Component loadings: features x components
    loadings = pd.DataFrame(
        result.pca_model.components_.T,
        index=feature_names or [f"feat_{i}" for i in range(result.pca_model.n_features_in_)],
        columns=[f"PC{i+1}" for i in range(result.pca_model.n_components_)],
    )
    loadings.to_parquet(loadings_path)

    return model_path
