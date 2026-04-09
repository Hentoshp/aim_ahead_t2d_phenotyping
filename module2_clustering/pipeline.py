"""Module 2 clustering pipeline.

Orchestrates PCA -> GMM grid search -> bootstrap stability -> profiling.
"""
from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path

import json
import pandas as pd

from module2_clustering.artifact_policy import resolve_module2_artifact_policy
from module2_clustering import bootstrap, cluster_profiling, dimensionality_reduction, gmm_clustering
from module2_clustering.utils import default_clustering_view, ensure_dir, load_config, resolve_paths, resolve_view_paths


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Module 2 clustering pipeline")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--skip-existing", action="store_true", help="Skip steps if artifacts already exist")
    parser.add_argument("--steps", default="pca,gmm,bootstrap,profiling", help="Comma-separated steps to run")
    parser.add_argument("--view", default=None, help="Optional clustering view name")
    parser.add_argument("--matrix-path", default=None, help="Optional explicit clustering matrix path")
    parser.add_argument("--matrix-meta-path", default=None, help="Optional explicit clustering matrix metadata path")
    parser.add_argument("--artifacts-dir", default=None, help="Optional explicit artifacts output directory")
    parser.add_argument("--experiment-name", default=None, help="Optional experiment name for view-scoped artifacts")
    return parser.parse_args()


def _merge_module2_overrides(cfg: dict, overrides: dict | None) -> dict:
    merged = copy.deepcopy(cfg)
    if overrides:
        merged.setdefault("module2", {}).update(copy.deepcopy(overrides))
    return merged


def run_pipeline(
    cfg_path: Path,
    steps: list[str] | None = None,
    view: str | None = None,
    matrix_path: Path | None = None,
    matrix_meta_path: Path | None = None,
    artifacts_path: Path | None = None,
    experiment_name: str | None = None,
    module2_overrides: dict | None = None,
) -> dict:
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = load_config(cfg_path)
    cfg = _merge_module2_overrides(cfg, module2_overrides)
    paths = resolve_paths(cfg)
    artifact_policy = resolve_module2_artifact_policy(cfg)
    selected_view = view or default_clustering_view(cfg)

    if view:
        view_paths = resolve_view_paths(cfg, view, experiment_name=experiment_name if artifacts_path is None else None)
        matrix_path = matrix_path or view_paths.clustering_matrix
        matrix_meta_path = matrix_meta_path or view_paths.clustering_meta
        artifacts_path = artifacts_path or view_paths.artifacts_path
    elif matrix_path is None:
        view_paths = resolve_view_paths(cfg, selected_view, experiment_name=experiment_name if artifacts_path is None else None)
        matrix_path = view_paths.clustering_matrix
        matrix_meta_path = matrix_meta_path or view_paths.clustering_meta
        artifacts_path = artifacts_path or view_paths.artifacts_path
    else:
        matrix_path = matrix_path or paths.clustering_matrix
        matrix_meta_path = matrix_meta_path or paths.clustering_meta
        artifacts_path = artifacts_path or paths.artifacts_path

    matrix_path = Path(matrix_path)
    matrix_meta_path = Path(matrix_meta_path) if matrix_meta_path else None
    artifacts_path = Path(artifacts_path)

    ensure_dir(artifacts_path)
    ensure_dir(paths.runs_path)

    logging.info("Loading clustering matrix from %s", matrix_path)
    if not matrix_path.exists():
        raise FileNotFoundError(f"clustering_matrix.parquet not found at {matrix_path}")

    if steps is None:
        steps = ["pca", "gmm", "bootstrap", "profiling"]

    matrix_meta = None
    if matrix_meta_path and matrix_meta_path.exists():
        matrix_meta = json.loads(matrix_meta_path.read_text())

    # 1) Load normalized clustering matrix
    matrix = pd.read_parquet(matrix_path)
    filtered_matrix = matrix
    dropped_corr: list[str] = []
    run_summary = {
        "view_name": selected_view,
        "experiment_name": experiment_name,
        "config": {
            "k_range": list(cfg["module2"]["k_range"]),
            "covariance_types": list(cfg["module2"]["covariance_types"]),
            "gmm_reg_covar": cfg["module2"].get("gmm_reg_covar", 1e-6),
            "pca_mode": cfg["module2"].get("pca_mode", "variance"),
            "pca_variance_threshold": cfg["module2"].get("pca_variance_threshold"),
            "pca_n_components": cfg["module2"].get("pca_n_components"),
            "corr_prune": cfg["module2"].get("corr_prune", False),
            "corr_threshold": cfg["module2"].get("corr_threshold", 0.9),
            "bootstrap_B": cfg["module2"]["bootstrap_B"],
            "bootstrap_early_stop_threshold": cfg["module2"]["bootstrap_early_stop_threshold"],
            "random_seed": cfg["module2"]["random_seed"],
        },
        "artifact_policy": {
            "level": artifact_policy.level,
            "save_plots": artifact_policy.save_plots,
            "save_pca_summary": artifact_policy.save_pca_summary,
            "save_pca_transformed": artifact_policy.save_pca_transformed,
            "save_pca_loadings": artifact_policy.save_pca_loadings,
            "save_pruned_matrix": artifact_policy.save_pruned_matrix,
            "save_debug_sidecars": artifact_policy.save_debug_sidecars,
            "save_json_mirrors": artifact_policy.save_json_mirrors,
            "compute_shap": artifact_policy.compute_shap,
            "save_shap_outputs": artifact_policy.save_shap_outputs,
        },
        "input": {
            "n_participants": int(matrix.shape[0]),
            "n_features_raw": int(matrix.shape[1]),
            "matrix_path": str(matrix_path),
            "matrix_meta_path": str(matrix_meta_path) if matrix_meta_path else None,
        },
    }
    if matrix_meta is not None:
        run_summary["input"]["matrix_meta"] = matrix_meta

    if cfg["module2"].get("corr_prune", False):
        filtered_matrix, dropped_corr = dimensionality_reduction.correlation_filter(
            matrix,
            threshold=cfg["module2"].get("corr_threshold", 0.9),
        )
        corr_summary = {
            "threshold": cfg["module2"].get("corr_threshold", 0.9),
            "dropped_features": dropped_corr,
            "remaining_features": list(filtered_matrix.columns),
        }
        if artifact_policy.save_debug_sidecars:
            _write_json(
                artifacts_path / "corr_pruned_features.json",
                corr_summary,
            )
        if artifact_policy.save_pruned_matrix:
            filtered_matrix.to_parquet(artifacts_path / "clustering_matrix_pruned.parquet")
        run_summary["correlation_pruning"] = corr_summary
    else:
        run_summary["correlation_pruning"] = {
            "threshold": None,
            "dropped_features": [],
            "remaining_features": list(filtered_matrix.columns),
        }

    run_summary["input"]["n_features_used"] = int(filtered_matrix.shape[1])

    # 2) PCA
    if "pca" in steps:
        pca_result = dimensionality_reduction.run_pca(
            filtered_matrix,
            variance_threshold=cfg["module2"]["pca_variance_threshold"],
            random_state=cfg["module2"]["random_seed"],
            artifacts_path=artifacts_path if artifact_policy.save_plots else None,
            mode=cfg["module2"].get("pca_mode", "variance"),
            n_components=cfg["module2"].get("pca_n_components"),
            save_plot=artifact_policy.save_plots,
        )
        dimensionality_reduction.save_pca_artifacts(
            pca_result,
            artifacts_path,
            feature_names=filtered_matrix.columns.tolist(),
            save_summary=artifact_policy.save_pca_summary,
            save_transformed=artifact_policy.save_pca_transformed,
            save_loadings=artifact_policy.save_pca_loadings,
            mode=cfg["module2"].get("pca_mode", "variance"),
            variance_threshold=cfg["module2"].get("pca_variance_threshold"),
            n_components_requested=cfg["module2"].get("pca_n_components"),
        )
    else:
        # load transformed if skipping
        trans_path = artifacts_path / "pca_transformed.parquet"
        model_path = artifacts_path / "pca_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError("PCA model artifact missing; cannot skip pca step.")
        import joblib
        pca_model = joblib.load(model_path)
        if trans_path.exists():
            transformed = pd.read_parquet(trans_path)
        else:
            transformed_arr = pca_model.transform(filtered_matrix.values)
            transformed = pd.DataFrame(
                transformed_arr,
                index=filtered_matrix.index,
                columns=[f"PC{i+1}" for i in range(transformed_arr.shape[1])],
            )
        explained = getattr(pca_model, "explained_variance_ratio_", None)
        explained_sum = float(explained.sum()) if explained is not None else None
        n_components = getattr(pca_model, "n_components_", None)
        pca_result = dimensionality_reduction.PCAResult(
            transformed=transformed,
            pca_model=pca_model,
            explained_variance=explained_sum,
            n_components=n_components,
        )

    run_summary["pca"] = {
        "mode": cfg["module2"].get("pca_mode", "variance"),
        "variance_threshold": cfg["module2"].get("pca_variance_threshold"),
        "n_components_requested": cfg["module2"].get("pca_n_components"),
        "n_components": int(pca_result.n_components) if pca_result.n_components is not None else None,
        "explained_variance": pca_result.explained_variance,
        "feature_names": filtered_matrix.columns.tolist(),
    }

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
        if artifact_policy.save_plots:
            gmm_clustering.plot_bic_curve(gmm_grid.grid, artifacts_path)
        gmm_grid.grid.to_csv(artifacts_path / "gmm_grid_search.csv", index=False)
        # base-fit membership diagnostics (no bootstrap averaging)
        base_membership = best_fit.model.predict_proba(pca_result.transformed.values)
        base_diag = gmm_clustering.membership_flags(base_membership, bootstrap_ari_std=None)
        if artifact_policy.save_debug_sidecars:
            _write_json(artifacts_path / "membership_diagnostics_base.json", base_diag)
        run_summary["gmm_selection"] = {
            "best_k": best_fit.k,
            "best_covariance_type": best_fit.covariance_type,
            "best_bic": best_fit.bic,
            "best_aic": best_fit.aic,
            "best_log_likelihood": best_fit.log_likelihood,
            "grid": gmm_grid.grid.to_dict(orient="records"),
        }
        run_summary["membership_diagnostics_base"] = base_diag
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
            compute_shap=artifact_policy.compute_shap,
            feature_names=pca_result.transformed.columns.tolist(),
            artifacts_path=artifacts_path if artifact_policy.save_shap_outputs else None,
        )
        # membership diagnostics
        diag = gmm_clustering.membership_flags(boot.membership_matrix, bootstrap_ari_std=boot.ari_std)
        if artifact_policy.save_debug_sidecars:
            _write_json(artifacts_path / "membership_diagnostics.json", diag)
        # bootstrap summary
        boot_summary = {
            "mean_ari": boot.mean_ari,
            "ci_ari": boot.ci_ari,
            "resamples_run": boot.resamples_run,
            "ari_std": boot.ari_std,
            "early_stopped": boot.early_stopped,
        }
        if artifact_policy.save_debug_sidecars:
            _write_json(artifacts_path / "bootstrap_summary.json", boot_summary)
        run_summary["membership_diagnostics_bootstrap_mean"] = diag
        run_summary["bootstrap"] = boot_summary
        if boot.shap_report is not None:
            run_summary["shap"] = boot.shap_report
    else:
        raise NotImplementedError("Skipping bootstrap not supported yet.")

    # 5) Profiling
    if "profiling" in steps:
        membership_df = cluster_profiling.build_membership_matrix(matrix.index, boot.membership_matrix, artifacts_path=artifacts_path)
        profiles_df = cluster_profiling.back_project_centroids(
            pca_result.pca_model,
            best_fit.model,
            feature_columns=filtered_matrix.columns.tolist(),
            artifacts_path=artifacts_path,
        )
        summary = cluster_profiling.summarize_profiles(membership_df, profiles_df)
        if artifact_policy.save_debug_sidecars:
            _write_json(artifacts_path / "cluster_profile_summary.json", summary)
        run_summary["cluster_profiles"] = summary

    _write_json(artifacts_path / "module2_run_summary.json", run_summary)

    logging.info("Module 2 pipeline completed.")
    return run_summary


def main():
    args = parse_args()
    cfg_path = Path(args.config)
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    run_pipeline(
        cfg_path=cfg_path,
        steps=steps,
        view=args.view,
        matrix_path=Path(args.matrix_path) if args.matrix_path else None,
        matrix_meta_path=Path(args.matrix_meta_path) if args.matrix_meta_path else None,
        artifacts_path=Path(args.artifacts_dir) if args.artifacts_dir else None,
        experiment_name=args.experiment_name,
    )


if __name__ == "__main__":
    main()
