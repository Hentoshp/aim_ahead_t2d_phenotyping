"""Ad-hoc diagnostics for dimensionality reduction + GMM.

Runs a small sweep over PCA settings (variance threshold and fixed
component counts) after an optional correlation filter. Saves summary
metrics to the artifacts directory (config-resolved), leaving the main
pipeline untouched.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from module2_clustering.utils import load_config, resolve_paths


@dataclass
class PCASpec:
    name: str
    mode: str  # "variance" or "n_components"
    value: float


def correlation_filter(df: pd.DataFrame, threshold: float = 0.9, method: str = "pearson") -> tuple[pd.DataFrame, list[str]]:
    """Drop columns that are highly correlated (|r| > threshold).

    Keeps the first occurrence in column order and drops later duplicates
    when a pair exceeds the threshold. Returns filtered df and dropped list.
    """

    if threshold <= 0 or threshold >= 1:
        raise ValueError("threshold must be in (0, 1)")
    if df.empty:
        return df.copy(), []

    corr = df.corr(method=method).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop: list[str] = [col for col in upper.columns if any(upper[col] > threshold)]
    filtered = df.drop(columns=to_drop)
    return filtered, to_drop


def _n_components_for_variance(X: np.ndarray, threshold: float) -> int:
    pca_full = PCA(random_state=None, svd_solver="full")
    pca_full.fit(X)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumvar, threshold) + 1)
    return max(1, min(n_components, X.shape[1]))


def _fit_pca(X: np.ndarray, spec: PCASpec, random_state: int | None) -> tuple[PCA, np.ndarray, float]:
    if spec.mode == "variance":
        n_components = _n_components_for_variance(X, spec.value)
    elif spec.mode == "n_components":
        n_components = int(spec.value)
    else:
        raise ValueError(f"Unknown PCA spec mode: {spec.mode}")

    pca = PCA(n_components=n_components, random_state=random_state, svd_solver="full")
    X_pca = pca.fit_transform(X)
    explained = float(pca.explained_variance_ratio_.sum())
    return pca, X_pca, explained


def _fit_gmm(X: np.ndarray, k: int, covariance: str, random_state: int | None) -> GaussianMixture:
    gmm = GaussianMixture(
        n_components=k,
        covariance_type=covariance,
        random_state=random_state,
        reg_covar=1e-6,
        max_iter=500,
    )
    gmm.fit(X)
    return gmm


def _entropy(probs: np.ndarray) -> tuple[float, float, float]:
    eps = 1e-12
    entropy = -np.sum(probs * np.log(probs + eps), axis=1)
    mean = float(entropy.mean())
    max_possible = float(np.log(probs.shape[1])) if probs.shape[1] else 0.0
    prop_high_conf = float((probs.max(axis=1) > 0.7).mean())
    return mean, max_possible, prop_high_conf


def run_diagnostics(
    df: pd.DataFrame,
    specs: Iterable[PCASpec],
    k: int,
    covariance: str,
    corr_threshold: float,
    random_state: int | None,
) -> pd.DataFrame:
    filtered_df, dropped = correlation_filter(df, threshold=corr_threshold)

    records: List[dict] = []
    for spec in specs:
        pca_model, X_pca, explained = _fit_pca(filtered_df.values, spec, random_state)
        gmm = _fit_gmm(X_pca, k=k, covariance=covariance, random_state=random_state)
        memberships = gmm.predict_proba(X_pca)
        ent_mean, ent_max, prop_conf = _entropy(memberships)
        records.append(
            {
                "spec": spec.name,
                "mode": spec.mode,
                "value": spec.value,
                "n_components": int(pca_model.n_components_),
                "explained_variance": explained,
                "bic": float(gmm.bic(X_pca)),
                "entropy_mean": ent_mean,
                "entropy_max": ent_max,
                "prop_high_confidence": prop_conf,
                "features_remaining": filtered_df.shape[1],
                "features_dropped_corr": len(dropped),
            }
        )

    return pd.DataFrame.from_records(records)


def default_specs() -> list[PCASpec]:
    return [
        PCASpec(name="var_0_75", mode="variance", value=0.75),
        PCASpec(name="var_0_80", mode="variance", value=0.80),
        PCASpec(name="var_0_88", mode="variance", value=0.88),
        PCASpec(name="n_6", mode="n_components", value=6),
        PCASpec(name="n_8", mode="n_components", value=8),
        PCASpec(name="n_10", mode="n_components", value=10),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCA/GMM diagnostics sweep")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument("--k", type=int, default=3, help="Number of GMM components")
    parser.add_argument("--covariance", default="full", help="GMM covariance_type")
    parser.add_argument("--corr-threshold", type=float, default=0.9, help="Correlation threshold for feature pruning")
    parser.add_argument(
        "--out",
        default=None,
        help="Output path (parquet). Defaults to <artifacts>/pca_gmm_diagnostics.parquet",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(Path(args.config))
    paths = resolve_paths(cfg)

    df = pd.read_parquet(paths.clustering_matrix)
    specs = default_specs()

    summary = run_diagnostics(
        df,
        specs=specs,
        k=args.k,
        covariance=args.covariance,
        corr_threshold=args.corr_threshold,
        random_state=cfg["module2"].get("random_seed"),
    )

    out_path = Path(args.out) if args.out else Path(paths.artifacts_path) / "pca_gmm_diagnostics.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(out_path, index=False)
    summary.to_csv(out_path.with_suffix(".csv"), index=False)

    # Small human-readable printout
    print(summary.sort_values("entropy_mean"))


if __name__ == "__main__":
    main()
