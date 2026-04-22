"""Promote a selected Module 2 experiment to a canonical slot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from module2_clustering.utils import ensure_dir, load_config, resolve_selected_paths, resolve_view_paths


REQUIRED_SOURCE_ARTIFACTS = [
    "module2_run_summary.json",
    "membership_matrix.parquet",
    "cluster_profiles.parquet",
    "pca_model.joblib",
    "gmm_grid_search.csv",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote a Module 2 experiment to a selected slot")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--slot", default="primary", help="Selected slot name, e.g. primary or sensitivity")
    parser.add_argument("--view", required=True, help="Source view name")
    parser.add_argument("--experiment", required=True, help="Source experiment name")
    parser.add_argument("--note", default=None, help="Optional free-text note recorded in the manifest")
    return parser.parse_args()


def promote_solution(cfg_path: Path, slot: str, view: str, experiment: str, note: str | None = None) -> Path:
    cfg = load_config(cfg_path)
    source_paths = resolve_view_paths(cfg, view, experiment_name=experiment)
    selected_paths = resolve_selected_paths(cfg, slot)

    missing = [name for name in REQUIRED_SOURCE_ARTIFACTS if not (source_paths.artifacts_path / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Cannot promote {view}/{experiment}; missing source artifacts: {missing}"
        )

    run_summary_path = source_paths.artifacts_path / "module2_run_summary.json"
    run_summary = json.loads(run_summary_path.read_text())

    ensure_dir(selected_paths.root)
    ensure_dir(selected_paths.shap_dir)

    manifest = {
        "slot": slot,
        "source_view": view,
        "source_experiment": experiment,
        "source_artifacts_path": str(source_paths.artifacts_path),
        "source_run_summary_path": str(run_summary_path),
        "required_artifacts": {
            name: str(source_paths.artifacts_path / name) for name in REQUIRED_SOURCE_ARTIFACTS
        },
        "selected_model": {
            "best_k": run_summary.get("gmm_selection", {}).get("best_k"),
            "best_covariance_type": run_summary.get("gmm_selection", {}).get("best_covariance_type"),
            "base_prop_high_confidence": run_summary.get("membership_diagnostics_base", {}).get("prop_high_confidence"),
            "bootstrap_mean_ari": run_summary.get("bootstrap", {}).get("mean_ari"),
            "smallest_hard_cluster_n": min(
                run_summary.get("cluster_profiles", {}).get("cluster_hard_sizes", {}).values(),
                default=None,
            ),
        },
        "note": note,
    }
    selected_paths.manifest_path.write_text(json.dumps(manifest, indent=2))
    return selected_paths.manifest_path


def main() -> None:
    args = _parse_args()
    manifest_path = promote_solution(
        cfg_path=Path(args.config),
        slot=args.slot,
        view=args.view,
        experiment=args.experiment,
        note=args.note,
    )
    print(f"Promoted solution manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
