"""Run final SHAP interpretation for a selected Module 2 solution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.mixture import GaussianMixture

from module2_clustering import shap_importance
from module2_clustering.utils import ensure_dir, load_config, resolve_selected_paths, resolve_view_paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SHAP on a selected Module 2 solution")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--slot", default=None, help="Selected slot name, e.g. primary or sensitivity")
    parser.add_argument("--view", default=None, help="Source view name when not using --slot")
    parser.add_argument("--experiment", default=None, help="Source experiment name when not using --slot")
    return parser.parse_args()


def _load_source_from_slot(cfg: dict, slot: str) -> tuple[str, str, Path, Path]:
    selected_paths = resolve_selected_paths(cfg, slot)
    if not selected_paths.manifest_path.exists():
        raise FileNotFoundError(f"Selection manifest not found: {selected_paths.manifest_path}")
    manifest = json.loads(selected_paths.manifest_path.read_text())
    view = manifest["source_view"]
    experiment = manifest["source_experiment"]
    source_artifacts = Path(manifest["source_artifacts_path"])
    out_dir = selected_paths.shap_dir
    return view, experiment, source_artifacts, out_dir


def _load_source_direct(cfg: dict, view: str, experiment: str) -> tuple[str, str, Path, Path]:
    source_paths = resolve_view_paths(cfg, view, experiment_name=experiment)
    return view, experiment, source_paths.artifacts_path, source_paths.artifacts_path / "shap_final"


def _top_features_per_cluster(summary_df: pd.DataFrame, top_n: int = 10) -> dict:
    output: dict[int, list[dict]] = {}
    for cluster, group in summary_df.groupby("cluster"):
        ranked = group.sort_values("abs_shap_mean", ascending=False)
        output[int(cluster)] = ranked.head(top_n)[["feature", "abs_shap_mean", "shap_mean"]].to_dict(orient="records")
    return output


def _load_or_refit_gmm(cfg: dict, run_summary: dict, source_artifacts: Path, X_pca) -> GaussianMixture:
    model_path = source_artifacts / "gmm_model.joblib"
    if model_path.exists():
        return joblib.load(model_path)

    module2_cfg = run_summary.get("config", {})
    gmm = GaussianMixture(
        n_components=int(run_summary["gmm_selection"]["best_k"]),
        covariance_type=str(run_summary["gmm_selection"]["best_covariance_type"]),
        reg_covar=float(module2_cfg.get("gmm_reg_covar", cfg["module2"].get("gmm_reg_covar", 1e-6))),
        random_state=int(module2_cfg.get("random_seed", cfg["module2"]["random_seed"])),
        max_iter=500,
    )
    gmm.fit(X_pca)
    return gmm


def run_shap(cfg_path: Path, slot: str | None = None, view: str | None = None, experiment: str | None = None) -> Path:
    cfg = load_config(cfg_path)
    if slot:
        resolved_view, resolved_experiment, source_artifacts, out_dir = _load_source_from_slot(cfg, slot)
    else:
        if not view or not experiment:
            raise ValueError("Provide either --slot or both --view and --experiment.")
        resolved_view, resolved_experiment, source_artifacts, out_dir = _load_source_direct(cfg, view, experiment)

    run_summary_path = source_artifacts / "module2_run_summary.json"
    if not run_summary_path.exists():
        raise FileNotFoundError(f"Missing run summary: {run_summary_path}")
    run_summary = json.loads(run_summary_path.read_text())

    matrix_path = Path(run_summary.get("input", {}).get("matrix_path") or resolve_view_paths(cfg, resolved_view).clustering_matrix)
    matrix = pd.read_parquet(matrix_path)
    feature_names = list(run_summary.get("correlation_pruning", {}).get("remaining_features", matrix.columns.tolist()))
    X = matrix.loc[:, feature_names]

    pca_model_path = source_artifacts / "pca_model.joblib"
    if not pca_model_path.exists():
        raise FileNotFoundError(f"Missing PCA model: {pca_model_path}")
    pca_model = joblib.load(pca_model_path)
    X_pca = pca_model.transform(X.values)

    gmm_model = _load_or_refit_gmm(cfg, run_summary, source_artifacts, X_pca)
    labels = gmm_model.predict(X_pca)
    memberships = gmm_model.predict_proba(X_pca)

    shap_result = shap_importance.compute_shap_distributions(
        X.values,
        labels,
        feature_names=feature_names,
        random_state=int(run_summary.get("config", {}).get("random_seed", cfg["module2"]["random_seed"])),
    )

    ensure_dir(out_dir)
    summary_path = out_dir / "shap_summary.parquet"
    report_path = out_dir / "shap_report.json"
    assignments_path = out_dir / "base_cluster_assignments.parquet"

    shap_result.summary.to_parquet(summary_path, index=False)

    assignment_df = pd.DataFrame(
        {
            "participant_id": X.index,
            "cluster": labels,
            "max_membership": memberships.max(axis=1),
        }
    ).set_index("participant_id")
    assignment_df.to_parquet(assignments_path)

    report = {
        "source_view": resolved_view,
        "source_experiment": resolved_experiment,
        "source_artifacts_path": str(source_artifacts),
        "feature_space": "original_pruned_features",
        "n_participants": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "label_source": "base_fit_gmm_hard_labels",
        "cluster_sizes": {str(int(cluster)): int((labels == cluster).sum()) for cluster in sorted(set(labels.tolist()))},
        "top_features_per_cluster": _top_features_per_cluster(shap_result.summary, top_n=10),
    }
    report_path.write_text(json.dumps(report, indent=2))
    return report_path


def main() -> None:
    args = _parse_args()
    report_path = run_shap(
        cfg_path=Path(args.config),
        slot=args.slot,
        view=args.view,
        experiment=args.experiment,
    )
    print(f"SHAP outputs written to {report_path.parent}")


if __name__ == "__main__":
    main()
