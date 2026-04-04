"""Module 2 clustering pipeline.

Orchestrates PCA → GMM grid search → bootstrap stability (with SHAP) → profiling.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import json
import pandas as pd

from module2_clustering import bootstrap, cluster_profiling, dimensionality_reduction, gmm_clustering, shap_importance
from module2_clustering.utils import ensure_dir, load_config, resolve_paths


def parse_args():
    parser = argparse.ArgumentParser(description="Module 2 clustering pipeline")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--skip-existing", action="store_true", help="Skip steps if artifacts already exist")
    parser.add_argument("--steps", default="pca,gmm,bootstrap,profiling", help="Comma-separated steps to run")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = load_config(cfg_path)
    paths = resolve_paths(cfg)

    ensure_dir(paths.artifacts_path)
    ensure_dir(paths.runs_path)

    logging.info("Loading clustering matrix from %s", paths.clustering_matrix)
    if not paths.clustering_matrix.exists():
        raise FileNotFoundError(f"clustering_matrix.parquet not found at {paths.clustering_matrix}")

    steps = [s.strip() for s in args.steps.split(",") if s.strip()]

    # 1) Load normalized clustering matrix
    matrix = pd.read_parquet(paths.clustering_matrix)
    filtered_matrix = matrix
    dropped_corr: list[str] = []

    if cfg["module2"].get("corr_prune", False):
        filtered_matrix, dropped_corr = dimensionality_reduction.correlation_filter(
            matrix,
            threshold=cfg["module2"].get("corr_threshold", 0.9),
        )
        (paths.artifacts_path / "corr_pruned_features.json").write_text(
            json.dumps(
                {
                    "threshold": cfg["module2"].get("corr_threshold", 0.9),
                    "dropped_features": dropped_corr,
                    "remaining_features": list(filtered_matrix.columns),
                },
                indent=2,
            )
        )
        # Save filtered matrix for reproducibility
        filtered_matrix.to_parquet(paths.artifacts_path / "clustering_matrix_pruned.parquet")

    # 2) PCA
    if "pca" in steps:
        pca_result = dimensionality_reduction.run_pca(
            filtered_matrix,
            variance_threshold=cfg["module2"]["pca_variance_threshold"],
            random_state=cfg["module2"]["random_seed"],
            artifacts_path=paths.artifacts_path,
            mode=cfg["module2"].get("pca_mode", "variance"),
            n_components=cfg["module2"].get("pca_n_components"),
        )
        dimensionality_reduction.save_pca_artifacts(
            pca_result,
            paths.artifacts_path,
            feature_names=filtered_matrix.columns.tolist(),
            save_transformed=True,
            mode=cfg["module2"].get("pca_mode", "variance"),
            variance_threshold=cfg["module2"].get("pca_variance_threshold"),
            n_components_requested=cfg["module2"].get("pca_n_components"),
        )
    else:
        # load transformed if skipping
        trans_path = paths.artifacts_path / "pca_transformed.parquet"
        model_path = paths.artifacts_path / "pca_model.joblib"
        if not trans_path.exists() or not model_path.exists():
            raise FileNotFoundError("PCA artifacts missing; cannot skip pca step.")
        import joblib
        pca_model = joblib.load(model_path)
        transformed = pd.read_parquet(trans_path)
        explained = getattr(pca_model, "explained_variance_ratio_", None)
        explained_sum = float(explained.sum()) if explained is not None else None
        n_components = getattr(pca_model, "n_components_", None)
        pca_result = dimensionality_reduction.PCAResult(
            transformed=transformed,
            pca_model=pca_model,
            explained_variance=explained_sum,
            n_components=n_components,
        )

    # 3) GMM grid search
    if "gmm" in steps:
        gmm_grid = gmm_clustering.grid_search_gmm(
            pca_result.transformed.values,
            k_range=cfg["module2"]["k_range"],
            covariance_types=cfg["module2"]["covariance_types"],
            random_state=cfg["module2"]["random_seed"],
            reg_covar=cfg["module2"].get("gmm_reg_covar", 1e-6),
        )
        best_fit = gmm_grid.best
        gmm_clustering.plot_bic_curve(gmm_grid.grid, paths.artifacts_path)
        # base-fit membership diagnostics (no bootstrap averaging)
        base_membership = best_fit.model.predict_proba(pca_result.transformed.values)
        base_diag = gmm_clustering.membership_flags(base_membership, bootstrap_ari_std=None)
        (paths.artifacts_path / "membership_diagnostics_base.json").write_text(pd.Series(base_diag).to_json(indent=2))
    else:
        raise NotImplementedError("Skipping gmm not supported yet.")

    # 4) Bootstrap stability (+ SHAP inside bootstrap)
    if "bootstrap" in steps:
        boot = bootstrap.bootstrap_stability(
            pca_result.transformed.values,
            best_fit,
            B=cfg["module2"]["bootstrap_B"],
            early_stop_threshold=cfg["module2"]["bootstrap_early_stop_threshold"],
            random_state=cfg["module2"]["random_seed"],
            n_jobs=-1,
            compute_shap=True,
            feature_names=pca_result.transformed.columns.tolist(),
            artifacts_path=paths.artifacts_path,
        )
        # membership diagnostics
        diag = gmm_clustering.membership_flags(boot.membership_matrix, bootstrap_ari_std=boot.ari_std)
        (paths.artifacts_path / "membership_diagnostics.json").write_text(pd.Series(diag).to_json(indent=2))
        # bootstrap summary
        boot_summary = {
            "mean_ari": boot.mean_ari,
            "ci_ari": boot.ci_ari,
            "resamples_run": boot.resamples_run,
            "ari_std": boot.ari_std,
            "early_stopped": boot.early_stopped,
        }
        (paths.artifacts_path / "bootstrap_summary.json").write_text(json.dumps(boot_summary, indent=2))
    else:
        raise NotImplementedError("Skipping bootstrap not supported yet.")

    # 5) Profiling
    if "profiling" in steps:
        membership_df = cluster_profiling.build_membership_matrix(matrix.index, boot.membership_matrix, artifacts_path=paths.artifacts_path)
        profiles_df = cluster_profiling.back_project_centroids(
            pca_result.pca_model,
            best_fit.model,
            feature_columns=filtered_matrix.columns.tolist(),
            artifacts_path=paths.artifacts_path,
        )
        summary = cluster_profiling.summarize_profiles(membership_df, profiles_df)
        (paths.artifacts_path / "cluster_profile_summary.json").write_text(pd.Series(summary).to_json(indent=2))

    logging.info("Module 2 pipeline completed.")


if __name__ == "__main__":
    main()
