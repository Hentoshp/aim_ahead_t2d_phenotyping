"""Run Module 2 clustering experiments across views and config overrides."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from module2_clustering.artifact_policy import resolve_module2_artifact_policy
from module2_clustering.pipeline import run_pipeline
from module2_clustering.utils import default_clustering_view, ensure_dir, load_config, resolve_paths


def _parse_csv_arg(value: str | None) -> list[str] | None:
    if not value:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def _experiment_specs(cfg: dict, selected_names: list[str] | None = None) -> list[dict]:
    exploration_cfg = cfg.get("module2", {}).get("exploration", {})
    experiments = exploration_cfg.get("experiments") or [{"name": "default"}]

    normalized = []
    for spec in experiments:
        name = str(spec.get("name", "default"))
        if selected_names and name not in selected_names:
            continue
        normalized.append({"name": name, "overrides": {k: v for k, v in spec.items() if k != "name"}})

    if selected_names:
        found = {spec["name"] for spec in normalized}
        missing = [name for name in selected_names if name not in found]
        if missing:
            raise ValueError(f"Requested experiment names not found in config: {missing}")

    return normalized


def _selected_views(cfg: dict, requested_views: list[str] | None = None) -> list[str]:
    if requested_views:
        return requested_views
    return list(cfg.get("module2", {}).get("exploration", {}).get("views", [default_clustering_view(cfg)]))


def _flatten_summary(summary: dict) -> dict:
    hard_sizes = summary.get("cluster_profiles", {}).get("cluster_hard_sizes", {})
    n_participants = int(summary.get("input", {}).get("n_participants", 0))
    largest_hard_cluster = max(hard_sizes.values()) if hard_sizes else None
    largest_hard_cluster_fraction = (largest_hard_cluster / n_participants) if largest_hard_cluster and n_participants else None
    smallest_hard_cluster = min(hard_sizes.values()) if hard_sizes else None
    smallest_hard_cluster_fraction = (smallest_hard_cluster / n_participants) if smallest_hard_cluster is not None and n_participants else None

    return {
        "view_name": summary.get("view_name"),
        "experiment_name": summary.get("experiment_name"),
        "n_participants": n_participants,
        "n_features_raw": summary.get("input", {}).get("n_features_raw"),
        "n_features_used": summary.get("input", {}).get("n_features_used"),
        "corr_dropped_n": len(summary.get("correlation_pruning", {}).get("dropped_features", [])),
        "pca_mode": summary.get("pca", {}).get("mode"),
        "pca_variance_threshold": summary.get("pca", {}).get("variance_threshold"),
        "pca_n_components": summary.get("pca", {}).get("n_components"),
        "best_k": summary.get("gmm_selection", {}).get("best_k"),
        "best_covariance_type": summary.get("gmm_selection", {}).get("best_covariance_type"),
        "best_bic": summary.get("gmm_selection", {}).get("best_bic"),
        "best_aic": summary.get("gmm_selection", {}).get("best_aic"),
        "best_log_likelihood": summary.get("gmm_selection", {}).get("best_log_likelihood"),
        "base_prop_high_confidence": summary.get("membership_diagnostics_base", {}).get("prop_high_confidence"),
        "base_mean_entropy": summary.get("membership_diagnostics_base", {}).get("mean_entropy"),
        "bootstrap_mean_ari": summary.get("bootstrap", {}).get("mean_ari"),
        "bootstrap_ari_std": summary.get("bootstrap", {}).get("ari_std"),
        "bootstrap_ci_low": (summary.get("bootstrap", {}).get("ci_ari") or [None, None])[0],
        "bootstrap_ci_high": (summary.get("bootstrap", {}).get("ci_ari") or [None, None])[1],
        "largest_hard_cluster_n": largest_hard_cluster,
        "largest_hard_cluster_fraction": largest_hard_cluster_fraction,
        "smallest_hard_cluster_n": smallest_hard_cluster,
        "smallest_hard_cluster_fraction": smallest_hard_cluster_fraction,
        "hard_cluster_sizes_json": json.dumps(hard_sizes, sort_keys=True),
    }


def _selection_params(cfg: dict) -> dict:
    defaults = {
        "min_base_prop_high_confidence": 0.70,
        "min_bootstrap_mean_ari": 0.50,
        "preferred_min_cluster_fraction": 0.10,
        "acceptable_min_cluster_fraction": 0.05,
        "view_priority": ["wearable_environment", "environment", "wearable"],
    }
    user_cfg = cfg.get("module2", {}).get("selection", {})
    params = defaults | user_cfg
    params["view_priority"] = list(params.get("view_priority", defaults["view_priority"]))
    return params


def _cluster_balance_band(smallest_fraction: float | None, params: dict) -> str:
    if smallest_fraction is None:
        return "unknown"
    if smallest_fraction >= params["preferred_min_cluster_fraction"]:
        return "preferred"
    if smallest_fraction >= params["acceptable_min_cluster_fraction"]:
        return "acceptable_with_caution"
    return "poor"


def _view_priority_rank(view_name: str | None, params: dict) -> int:
    priorities = params["view_priority"]
    if view_name in priorities:
        return priorities.index(view_name)
    return len(priorities)


def _apply_selection_rule(comparison_df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = comparison_df.copy()
    df["min_base_prop_high_confidence_threshold"] = params["min_base_prop_high_confidence"]
    df["min_bootstrap_mean_ari_threshold"] = params["min_bootstrap_mean_ari"]
    df["preferred_min_cluster_fraction_threshold"] = params["preferred_min_cluster_fraction"]
    df["acceptable_min_cluster_fraction_threshold"] = params["acceptable_min_cluster_fraction"]
    df["passes_base_confidence"] = df["base_prop_high_confidence"] >= params["min_base_prop_high_confidence"]
    df["passes_bootstrap_stability"] = df["bootstrap_mean_ari"] >= params["min_bootstrap_mean_ari"]
    df["passes_minimum_viability"] = df["passes_base_confidence"] & df["passes_bootstrap_stability"]
    df["cluster_balance_band"] = df["smallest_hard_cluster_fraction"].apply(lambda x: _cluster_balance_band(x, params))
    df["passes_downstream_usability"] = df["cluster_balance_band"].isin(["preferred", "acceptable_with_caution"])
    df["view_priority_rank"] = df["view_name"].apply(lambda x: _view_priority_rank(x, params))

    def _status(row: pd.Series) -> str:
        if not row["passes_minimum_viability"]:
            return "reject"
        if row["cluster_balance_band"] == "preferred":
            return "finalist"
        if row["cluster_balance_band"] == "acceptable_with_caution":
            return "finalist_with_caution"
        return "technically_strong_but_downstream_risky"

    df["selection_status"] = df.apply(_status, axis=1)

    sort_cols = [
        "passes_minimum_viability",
        "passes_downstream_usability",
        "bootstrap_mean_ari",
        "base_prop_high_confidence",
        "smallest_hard_cluster_fraction",
    ]
    ascending = [False, False, False, False, False]
    df = df.sort_values(by=sort_cols + ["view_priority_rank", "bootstrap_ari_std"], ascending=ascending + [True, True])
    return df


def run_experiments(cfg_path: Path, views: list[str] | None = None, experiment_names: list[str] | None = None, steps: list[str] | None = None) -> pd.DataFrame:
    cfg = load_config(cfg_path)
    paths = resolve_paths(cfg)
    artifact_policy = resolve_module2_artifact_policy(cfg)
    module2_root = paths.artifacts_path / "module2"
    ensure_dir(module2_root)

    selected_views = _selected_views(cfg, requested_views=views)
    experiment_specs = _experiment_specs(cfg, selected_names=experiment_names)

    summaries: list[dict] = []
    for experiment in experiment_specs:
        experiment_name = experiment["name"]
        for view_name in selected_views:
            view_artifacts = module2_root / view_name / experiment_name
            summary = run_pipeline(
                cfg_path=cfg_path,
                steps=steps,
                view=view_name,
                artifacts_path=view_artifacts,
                experiment_name=experiment_name,
                module2_overrides=experiment["overrides"],
            )
            summaries.append(summary)

    comparison_df = pd.DataFrame([_flatten_summary(summary) for summary in summaries])
    selection_params = _selection_params(cfg)
    selection_df = _apply_selection_rule(comparison_df, selection_params)
    comparison_df.to_csv(module2_root / "experiment_comparison.csv", index=False)
    selection_df.to_csv(module2_root / "selection_summary.csv", index=False)
    if artifact_policy.save_json_mirrors:
        (module2_root / "experiment_comparison.json").write_text(json.dumps(summaries, indent=2, default=str))
        (module2_root / "selection_summary.json").write_text(
            json.dumps(
                {
                    "selection_parameters": selection_params,
                    "candidates": selection_df.to_dict(orient="records"),
                },
                indent=2,
                default=str,
            )
        )
    return selection_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Module 2 clustering experiments across views")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--views", default=None, help="Optional comma-separated subset of views")
    parser.add_argument("--experiments", default=None, help="Optional comma-separated subset of experiment names")
    parser.add_argument("--steps", default="pca,gmm,bootstrap,profiling", help="Comma-separated steps to run")
    args = parser.parse_args()

    comparison_df = run_experiments(
        cfg_path=Path(args.config),
        views=_parse_csv_arg(args.views),
        experiment_names=_parse_csv_arg(args.experiments),
        steps=[step.strip() for step in args.steps.split(",") if step.strip()],
    )
    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()
