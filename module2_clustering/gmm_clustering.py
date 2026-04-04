"""Gaussian Mixture Model utilities (scaffolding)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
from sklearn.mixture import GaussianMixture
import pandas as pd
from pathlib import Path
import math
import matplotlib.pyplot as plt


@dataclass
class GMMFit:
    model: GaussianMixture
    k: int
    covariance_type: str
    bic: float
    aic: float
    log_likelihood: float


@dataclass
class GMMGridResult:
    best: GMMFit
    grid: pd.DataFrame  # columns: k, covariance_type, bic, aic, log_likelihood


def grid_search_gmm(
    X: np.ndarray,
    k_range: Iterable[int],
    covariance_types: Iterable[str],
    random_state: int | None = None,
    reg_covar: float = 1e-6,
) -> GMMGridResult:
    """Run grid search over K and covariance type, select lowest BIC; return best fit and the full grid."""

    if X is None or len(X) == 0:
        raise ValueError("Input matrix X is empty.")
    if not np.isfinite(X).all():
        raise ValueError("Input matrix X contains non-finite values.")

    best: GMMFit | None = None
    records: List[dict] = []

    for k in k_range:
        if k < 1:
            continue
        for cov in covariance_types:
            try:
                gmm = GaussianMixture(
                    n_components=int(k),
                    covariance_type=str(cov),
                    random_state=random_state,
                    reg_covar=reg_covar,
                    max_iter=500,
                )
                gmm.fit(X)
                bic = gmm.bic(X)
                aic = gmm.aic(X)
                ll = float(gmm.score(X) * len(X))  # total log-likelihood
                fit = GMMFit(model=gmm, k=int(k), covariance_type=str(cov), bic=float(bic), aic=float(aic), log_likelihood=ll)
                records.append({"k": int(k), "covariance_type": str(cov), "bic": float(bic), "aic": float(aic), "log_likelihood": ll})

                if best is None:
                    best = fit
                else:
                    if fit.bic < best.bic - 1e-6:
                        best = fit
                    elif abs(fit.bic - best.bic) <= 1e-6:
                        # tie-breaker: higher log-likelihood, then smaller k
                        if fit.log_likelihood > best.log_likelihood + 1e-6 or (
                            abs(fit.log_likelihood - best.log_likelihood) <= 1e-6 and fit.k < best.k
                        ):
                            best = fit
            except Exception:
                continue

    if best is None:
        raise RuntimeError("All GMM fits failed; check data or parameter ranges.")

    grid_df = pd.DataFrame.from_records(records)
    return GMMGridResult(best=best, grid=grid_df)


def predict_membership(fit: GMMFit, X: np.ndarray) -> np.ndarray:
    """Return soft membership probabilities for samples."""

    if fit is None or fit.model is None:
        raise ValueError("GMMFit/model is None.")
    if X is None or len(X) == 0:
        raise ValueError("Input matrix X is empty.")
    if not np.isfinite(X).all():
        raise ValueError("Input matrix X contains non-finite values.")

    probs = fit.model.predict_proba(X)
    if probs.shape[1] != fit.k:
        raise ValueError("Predicted membership columns do not match k.")
    return probs


def plot_bic_curve(grid: pd.DataFrame, artifacts_path: Path) -> Path:
    """Plot BIC vs K (best per K) and save to artifacts."""
    artifacts_path.mkdir(parents=True, exist_ok=True)
    if grid.empty:
        raise ValueError("Grid is empty; cannot plot BIC.")
    best_per_k = grid.groupby("k")["bic"].min().reset_index()
    out_path = artifacts_path / "gmm_bic_curve.png"
    plt.figure(figsize=(6, 4))
    plt.plot(best_per_k["k"], best_per_k["bic"], marker="o")
    plt.xlabel("K")
    plt.ylabel("BIC (lower better)")
    plt.title("GMM BIC by K (best covariance per K)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def membership_flags(memberships: np.ndarray, bootstrap_ari_std: float | None = None) -> dict:
    """Compute diagnostic flags for soft memberships."""
    if memberships is None or memberships.size == 0:
        raise ValueError("Membership matrix is empty.")
    if not np.isfinite(memberships).all():
        raise ValueError("Membership matrix contains non-finite values.")

    max_probs = memberships.max(axis=1)
    prop_high_conf = float((max_probs > 0.7).mean()) if len(max_probs) else 0.0

    # entropy per participant
    eps = 1e-12
    entropy = -np.sum(memberships * np.log(memberships + eps), axis=1)
    k = memberships.shape[1]
    mean_entropy = float(entropy.mean()) if len(entropy) else 0.0
    max_entropy = math.log(k) if k > 0 else 0.0

    flags = []
    if prop_high_conf < 0.5:
        flags.append("Fewer than 50% of participants have max membership > 0.7")
    if max_entropy > 0 and mean_entropy > 0.8 * max_entropy:
        flags.append("Mean cluster entropy near maximum — assignments near-uniform")
    if bootstrap_ari_std is not None and bootstrap_ari_std > 0.15:
        flags.append("High bootstrap ARI variance — cluster structure may be unstable")

    return {
        "prop_high_confidence": prop_high_conf,
        "mean_entropy": mean_entropy,
        "max_entropy": max_entropy,
        "bootstrap_ari_std": bootstrap_ari_std,
        "flags": flags,
    }
