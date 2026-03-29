"""Module 2 clustering pipeline scaffolding.

Orchestrates PCA → GMM grid search → bootstrap stability → SHAP → profiling.
Implementation will be added once input artifacts are available.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from module2_clustering import bootstrap, cluster_profiling, dimensionality_reduction, gmm_clustering, shap_importance
from module2_clustering.utils import ensure_dir, load_config, resolve_paths


def parse_args():
    parser = argparse.ArgumentParser(description="Module 2 clustering pipeline")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
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

    # NOTE: actual processing is intentionally left unimplemented for now.
    # The outline below documents the intended call order.

    # 1) Load normalized clustering matrix
    matrix = pd.read_parquet(paths.clustering_matrix)

    # 2) Dimensionality reduction (PCA)
    # pca_result = dimensionality_reduction.run_pca(
    #     matrix,
    #     variance_threshold=cfg["module2"]["pca_variance_threshold"],
    #     random_state=cfg["module2"]["random_seed"],
    # )
    # dimensionality_reduction.save_pca_artifacts(pca_result, paths.artifacts_path)

    # 3) GMM grid search on PCA space
    # gmm_best = gmm_clustering.grid_search_gmm(
    #     pca_result.transformed,
    #     k_range=cfg["module2"]["k_range"],
    #     covariance_types=cfg["module2"]["covariance_types"],
    #     random_state=cfg["module2"]["random_seed"],
    # )

    # 4) Bootstrap stability & membership aggregation
    # boot = bootstrap.bootstrap_stability(
    #     pca_result.transformed,
    #     gmm_best.model,
    #     B=cfg["module2"]["bootstrap_B"],
    #     early_stop_threshold=cfg["module2"]["bootstrap_early_stop_threshold"],
    #     random_state=cfg["module2"]["random_seed"],
    # )

    # 5) Cluster profiling (memberships + back-projected centroids)
    # membership_df = cluster_profiling.build_membership_matrix(matrix.index, boot.membership_matrix)
    # profiles_df = cluster_profiling.back_project_centroids(
    #     pca_result.pca_model,
    #     gmm_best.model,
    #     feature_columns=matrix.columns.tolist(),
    # )

    # 6) SHAP importance per feature/cluster
    # shap_res = shap_importance.compute_shap_distributions(
    #     pca_result.transformed,
    #     gmm_best.model.predict(pca_result.transformed),
    #     feature_names=matrix.columns.tolist(),
    #     random_state=cfg["module2"]["random_seed"],
    # )

    raise NotImplementedError(
        "Module 2 pipeline is scaffolded; fill in PCA → GMM → bootstrap → SHAP → profiling implementations."
    )


if __name__ == "__main__":
    main()
