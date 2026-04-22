from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from module2_clustering.utils import ensure_dir, load_config, resolve_paths, resolve_selected_paths


FEATURE_LABELS = {
    "heart_rate_median": "Heart rate",
    "heart_rate_iqr": "Heart rate variability",
    "oxygen_sat_median": "Oxygen saturation",
    "oxygen_sat_iqr": "Oxygen saturation variability",
    "physical_activity_median": "Physical activity",
    "physical_activity_iqr": "Physical activity variability",
    "calories_median": "Calories",
    "calories_iqr": "Calories variability",
    "respiratory_rate_median": "Respiratory rate",
    "respiratory_rate_iqr": "Respiratory rate variability",
    "stress_median": "Stress",
    "stress_iqr": "Stress variability",
    "sleep_total_median_hr": "Total sleep",
    "sleep_total_iqr_hr": "Sleep variability",
    "sleep_deep_rem_median_hr": "Deep + REM sleep",
    "heart_rate_resting_median": "Resting heart rate",
    "oxygen_sat_prop_below_95": "O2 <95%",
    "env_pm10_median": "PM10",
    "env_hum_median": "Humidity",
    "env_hum_iqr": "Humidity variability",
    "env_temp_median": "Temperature",
    "env_temp_iqr": "Temperature variability",
    "env_voc_median": "VOC",
    "env_voc_iqr": "VOC variability",
    "env_voc_prop_high": "VOC high proportion",
    "env_nox_prop_high": "NOx high proportion",
    "env_light_total_median": "Light exposure",
    "env_light_total_iqr": "Light variability",
    "env_light_total_prop_high": "High light proportion",
}


@dataclass
class PosterContext:
    cfg: dict
    slot: str
    poster_root: Path
    plots_dir: Path
    tables_dir: Path
    manifest: dict
    run_summary: dict
    profiles_df: pd.DataFrame
    shap_summary_df: pd.DataFrame
    shap_report: dict
    assignments_df: pd.DataFrame
    selection_df: pd.DataFrame


def display_feature_name(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " "))


def cluster_display_name(cluster: int) -> str:
    return f"C{int(cluster) + 1}"


def build_profile_matrix(profiles_df: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    feature_order = (
        profiles_df.groupby("feature")["centroid_value"]
        .apply(lambda s: float(np.abs(s).max()))
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )
    matrix = (
        profiles_df[profiles_df["feature"].isin(feature_order)]
        .pivot(index="cluster", columns="feature", values="centroid_value")
        .reindex(sorted(profiles_df["cluster"].unique()))
        .loc[:, feature_order]
    )
    return matrix


def load_poster_context(cfg_path: Path, slot: str = "primary") -> PosterContext:
    cfg = load_config(cfg_path)
    selected_paths = resolve_selected_paths(cfg, slot)
    if not selected_paths.manifest_path.exists():
        raise FileNotFoundError(f"Selection manifest not found: {selected_paths.manifest_path}")

    manifest = json.loads(selected_paths.manifest_path.read_text())
    source_root = Path(manifest["source_artifacts_path"])
    run_summary = json.loads((source_root / "module2_run_summary.json").read_text())
    profiles_df = pd.read_parquet(source_root / "cluster_profiles.parquet")
    shap_summary_df = pd.read_parquet(selected_paths.shap_dir / "shap_summary.parquet")
    shap_report = json.loads((selected_paths.shap_dir / "shap_report.json").read_text())
    assignments_df = pd.read_parquet(selected_paths.shap_dir / "base_cluster_assignments.parquet")

    paths = resolve_paths(cfg)
    selection_path = paths.artifacts_path / "module2" / "selection_summary.csv"
    selection_df = pd.read_csv(selection_path)

    poster_root = paths.artifacts_path / "poster" / slot
    plots_dir = poster_root / "plots"
    tables_dir = poster_root / "tables"
    ensure_dir(plots_dir)
    ensure_dir(tables_dir)

    return PosterContext(
        cfg=cfg,
        slot=slot,
        poster_root=poster_root,
        plots_dir=plots_dir,
        tables_dir=tables_dir,
        manifest=manifest,
        run_summary=run_summary,
        profiles_df=profiles_df,
        shap_summary_df=shap_summary_df,
        shap_report=shap_report,
        assignments_df=assignments_df,
        selection_df=selection_df,
    )


def load_selected_feature_matrix(ctx: PosterContext) -> pd.DataFrame:
    matrix_path = Path(ctx.run_summary["input"]["matrix_path"])
    matrix = pd.read_parquet(matrix_path)
    feature_names = list(ctx.run_summary.get("correlation_pruning", {}).get("remaining_features", matrix.columns.tolist()))
    return matrix.loc[:, feature_names].copy()


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    header = "| " + " | ".join(df.columns.astype(str)) + " |"
    sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = []
    for _, row in df.iterrows():
        values = [str(value) for value in row.tolist()]
        rows.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join([header, sep, *rows]) + "\n")
