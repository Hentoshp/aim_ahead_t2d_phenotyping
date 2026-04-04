"""Bootstrap stability loop for Module 2."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from pathlib import Path
import math

from module2_clustering.gmm_clustering import GMMFit
from module2_clustering import shap_importance


@dataclass
class BootstrapResult:
    mean_ari: float
    ci_ari: Tuple[float, float]
    resamples_run: int
    early_stopped: bool
    membership_matrix: np.ndarray | None = None
    ari_std: float | None = None
    shap_summary: Optional[pd.DataFrame] = None
    shap_report: Optional[dict] = None


def bootstrap_stability(
    X: np.ndarray,
    base_gmm: GMMFit,
    B: int,
    early_stop_threshold: float,
    random_state: int | None = None,
    n_jobs: int = -1,
    compute_shap: bool = False,
    feature_names: Optional[List[str]] = None,
    artifacts_path: Optional[Path] = None,
) -> BootstrapResult:
    """Run bootstrap stability over fitted GMM.

    - Resample rows with replacement, refit GMM with same K/covariance.
    - Compute ARI vs base labels.
    - Average membership probabilities over successful resamples.
    - Optionally compute SHAP summaries per resample and aggregate.
    """

    if X is None or len(X) == 0:
        raise ValueError("Input matrix X is empty.")
    if not np.isfinite(X).all():
        raise ValueError("Input matrix X contains non-finite values.")
    if B < 1:
        raise ValueError("B must be >= 1.")
    if base_gmm is None or base_gmm.model is None:
        raise ValueError("base_gmm must be a fitted GMMFit.")

    rng = np.random.default_rng(random_state)
    base_labels = base_gmm.model.predict(X)
    n, _ = X.shape
    k = base_gmm.k

    seeds = rng.integers(0, 1_000_000_000, size=B)

    def _one(seed):
        rs = int(seed)
        idx = np.random.default_rng(rs).integers(0, n, size=n)
        Xb = X[idx]
        try:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type=base_gmm.covariance_type,
                reg_covar=getattr(base_gmm.model, "reg_covar", 1e-6) or 1e-6,
                max_iter=base_gmm.model.max_iter,
                random_state=rs,
            )
            gmm.fit(Xb)
            labels_b = gmm.predict(Xb)
            ari = adjusted_rand_score(base_labels[idx], labels_b)
            memberships_full = gmm.predict_proba(X)

            shap_summary = None
            if compute_shap:
                labels_full = gmm.predict(X)
                try:
                    shap_res = shap_importance.compute_shap_distributions(
                        X,
                        labels_full,
                        feature_names=feature_names or [f"feat_{i}" for i in range(X.shape[1])],
                        random_state=rs,
                    )
                    shap_summary = shap_res.summary.assign(resample_seed=rs)
                except Exception as e:
                    shap_summary = None

            return {"ari": ari, "membership": memberships_full, "shap_summary": shap_summary}
        except Exception as e:
            return {"error": repr(e)}

    results = Parallel(n_jobs=n_jobs, verbose=0, prefer="processes")(delayed(_one)(s) for s in seeds)
    errors = [r for r in results if r is not None and "error" in r]
    results = [r for r in results if r is not None and "error" not in r]

    if not results:
        msg = errors[0]["error"] if errors else "unknown"
        raise RuntimeError(f"All bootstrap resamples failed. Example error: {msg}")

    aris = np.array([r["ari"] for r in results])
    resamples_run = len(aris)
    ari_mean = float(aris.mean())
    ari_ci = (float(np.quantile(aris, 0.025)), float(np.quantile(aris, 0.975)))
    ari_std = float(np.std(aris))

    memberships_stack = np.stack([r["membership"] for r in results], axis=0)
    membership_mean = memberships_stack.mean(axis=0)

    early_stopped = False
    if resamples_run >= 20 and early_stop_threshold is not None and early_stop_threshold > 0:
        window = min(50, resamples_run)
        recent_std = float(np.std(aris[-window:]))
        if recent_std < early_stop_threshold:
            early_stopped = True

    shap_summary_agg = None
    shap_report = None
    if compute_shap:
        shap_frames = [r["shap_summary"] for r in results if r["shap_summary"] is not None]
        if shap_frames:
            all_shap = pd.concat(shap_frames, ignore_index=True)
            agg = (
                all_shap.groupby(["cluster", "feature"])["abs_shap_mean"]
                .agg([
                    ("mean_abs_shap", "mean"),
                    ("abs_shap_p2_5", lambda x: np.quantile(x, 0.025)),
                    ("abs_shap_p97_5", lambda x: np.quantile(x, 0.975)),
                ])
                .reset_index()
            )
            shap_summary_agg = agg
            if artifacts_path is not None:
                artifacts_path.mkdir(parents=True, exist_ok=True)
                agg.to_parquet(artifacts_path / "shap_distributions.parquet", index=False)

            # Simple validation/report payload
            shap_report = {
                "resamples_with_shap": len(shap_frames),
                "top_features_per_cluster": _top_features_per_cluster(agg, top_n=5),
            }
            if artifacts_path is not None:
                (artifacts_path / "shap_report.json").write_text(pd.Series(shap_report).to_json(indent=2))

    else:
        shap_report = None

    return BootstrapResult(
        mean_ari=ari_mean,
        ci_ari=ari_ci,
        resamples_run=resamples_run,
        early_stopped=early_stopped,
        membership_matrix=membership_mean,
        ari_std=ari_std,
        shap_summary=shap_summary_agg,
        shap_report=shap_report,
    )


def _top_features_per_cluster(shap_agg: pd.DataFrame, top_n: int = 5) -> dict:
    out: dict = {}
    for c, dfc in shap_agg.groupby("cluster"):
        dfc = dfc.sort_values("mean_abs_shap", ascending=False)
        out[int(c)] = dfc.head(top_n)[["feature", "mean_abs_shap", "abs_shap_p2_5", "abs_shap_p97_5"]].to_dict(orient="records")
    return out
