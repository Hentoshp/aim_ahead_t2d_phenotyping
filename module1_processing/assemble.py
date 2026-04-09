from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from .common import load_config, ensure_dirs

MODALITIES = [
    "wearable",
    "environment",
    "cgm",
    "clinical",
]

WEARABLE_CLUSTERING_PREFIXES = (
    "heart_rate_",
    "oxygen_sat_",
    "physical_activity_",
    "calories_",
    "respiratory_rate_",
    "stress_",
    "sleep_",
)

ENVIRONMENT_CLUSTERING_PREFIXES = (
    "env_",
)

CLUSTERING_PREFIXES = WEARABLE_CLUSTERING_PREFIXES + ENVIRONMENT_CLUSTERING_PREFIXES


def _is_clustering_feature(col: str) -> bool:
    if not col.startswith(CLUSTERING_PREFIXES):
        return False
    # drop coverage/meta fields
    if col.endswith("_valid_hours") or col.endswith("_ndays"):
        return False
    # drop sleep unknown stage metrics
    if "sleep_unknown" in col:
        return False
    return True


def _load_clustering_views(cfg: dict) -> tuple[str, str, dict[str, tuple[str, ...]]]:
    view_cfg = cfg.get("module1", {}).get("clustering_views", {})
    cohort_policy = str(view_cfg.get("cohort_policy", "common"))
    if cohort_policy != "common":
        raise NotImplementedError("Only cohort_policy='common' is supported.")

    default_views = {
        "wearable": WEARABLE_CLUSTERING_PREFIXES,
        "environment": ENVIRONMENT_CLUSTERING_PREFIXES,
        "wearable_environment": CLUSTERING_PREFIXES,
    }
    raw_views = view_cfg.get("views") or {
        name: {"include_prefixes": list(prefixes)}
        for name, prefixes in default_views.items()
    }

    views: dict[str, tuple[str, ...]] = {}
    for view_name, spec in raw_views.items():
        prefixes = tuple(spec.get("include_prefixes", []))
        if not prefixes:
            raise ValueError(f"Clustering view '{view_name}' must define include_prefixes.")
        views[str(view_name)] = prefixes

    default_view = str(view_cfg.get("default_view", "wearable_environment"))
    if default_view not in views:
        raise ValueError(f"default_view '{default_view}' not found in module1.clustering_views.views")

    return cohort_policy, default_view, views


def _select_view_columns(columns: list[str], include_prefixes: tuple[str, ...]) -> list[str]:
    selected = []
    for col in columns:
        if not _is_clustering_feature(col):
            continue
        if any(col.startswith(prefix) for prefix in include_prefixes):
            selected.append(col)
    return selected


def _module1_artifact_policy(cfg: dict) -> dict:
    raw_cfg = cfg.get("module1", {}).get("artifacts", {})
    level = str(raw_cfg.get("level", "standard"))
    defaults = {
        "standard": {
            "write_default_aliases": False,
            "save_view_raw_matrices": True,
            "save_common_raw_matrix": False,
        },
        "debug": {
            "write_default_aliases": True,
            "save_view_raw_matrices": True,
            "save_common_raw_matrix": True,
        },
    }
    if level not in defaults:
        raise ValueError(f"Unknown module1 artifact level: {level}")
    return {"level": level, **(defaults[level] | {k: v for k, v in raw_cfg.items() if k != "level"})}


def assemble(cfg_path: Path) -> None:
    cfg, base = load_config(cfg_path)
    artifact_policy = _module1_artifact_policy(cfg)
    inter_dir = Path(cfg["data"]["intermediates_path"].replace("${AIREADI_DATA_PATH}", str(base)))
    processed_path = Path(cfg["data"]["processed_path"].replace("${AIREADI_DATA_PATH}", str(base)))
    views_root = processed_path / "clustering_views"

    ensure_dirs(processed_path, views_root)

    clinical_df_full = pd.read_parquet(inter_dir / "clinical_features.parquet")
    pre_stage_counts = clinical_df_full["diabetes_stage"].value_counts(dropna=False).to_dict()

    dfs = []
    drop_log = {}
    for modality in MODALITIES:
        path = inter_dir / f"{modality}_features.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing intermediate: {path}")
        df = pd.read_parquet(path)
        dfs.append((modality, df))

    joined = dfs[0][1]
    for modality, df in dfs[1:]:
        before = len(joined)
        joined = joined.join(df, how="inner")
        dropped = before - len(joined)
        drop_log[modality] = int(dropped)

    if joined.empty:
        raise ValueError("Assembly produced empty feature matrix; check upstream modality outputs and QC thresholds")

    cohort_policy, default_view, clustering_views = _load_clustering_views(cfg)

    clustering_cols = [c for c in joined.columns if _is_clustering_feature(c)]
    outcome_cols = [c for c in [
        "glycemic_cv", "mean_glucose", "time_in_range",
        "hba1c", "hba1c_stratum", "diabetes_stage"
    ] if c in joined.columns]

    clustering_df = joined[clustering_cols].copy()
    outcome_df = joined[outcome_cols].copy()

    # Right-skewed features → log1p transform before normalization
    skew_prefixes = (
        "calories_",
        "respiratory_rate_",
        "env_pm1_",
        "env_pm2.5_",
        "env_pm10_",
        "env_light_total_",
    )
    skew_contains = ("sleep_total",)
    skew_cols = [
        c for c in clustering_df.columns
        if (c.startswith(skew_prefixes) or any(s in c for s in skew_contains))
        and not c.endswith("_prop_high")
    ]
    for col in skew_cols:
        clustering_df[col] = np.log1p(clustering_df[col].clip(lower=0))

    # Handle missing values before normalization/clustering
    missing_strategy = cfg["module1"].get("missing_strategy", "drop")
    if missing_strategy == "drop":
        before = len(clustering_df)
        mask = clustering_df.notna().all(axis=1)
        clustering_df = clustering_df.loc[mask]
        outcome_df = outcome_df.loc[mask]  # keep alignment
        drop_log["missing_values"] = int(before - len(clustering_df))
    elif missing_strategy == "impute":
        raise NotImplementedError("missing_strategy 'impute' not implemented yet; use 'drop' or 'none'")
    elif missing_strategy == "none":
        pass
    else:
        raise ValueError(f"Unknown missing_strategy: {missing_strategy}")

    # Save unscaled, transformed common-cohort matrix for diagnostics and view derivation.
    clustering_df_raw = clustering_df.copy()
    if artifact_policy["save_common_raw_matrix"]:
        common_raw_path = processed_path / "clustering_matrix_common_raw.parquet"
        clustering_df_raw.to_parquet(common_raw_path)

    if cfg["module1"].get("normalization") == "standard_scaler" and not clustering_df.empty:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(clustering_df.astype(float))
        clustering_df = pd.DataFrame(scaled, index=clustering_df.index, columns=clustering_df.columns)
        norm_meta = {"method": "StandardScaler", "with_mean": True, "with_std": True}
    else:
        norm_meta = {"method": None}

    default_scaled_df: pd.DataFrame | None = None
    default_raw_df: pd.DataFrame | None = None
    default_view_meta: dict | None = None

    for view_name, include_prefixes in clustering_views.items():
        view_cols = _select_view_columns(clustering_df.columns.tolist(), include_prefixes)
        if not view_cols:
            raise ValueError(f"Clustering view '{view_name}' selected zero features.")

        scaled_view_df = clustering_df.loc[:, view_cols].copy()
        raw_view_df = clustering_df_raw.loc[:, view_cols].copy()
        view_dir = views_root / view_name
        ensure_dirs(view_dir)

        scaled_view_df.to_parquet(view_dir / "clustering_matrix.parquet")
        if artifact_policy["save_view_raw_matrices"]:
            raw_view_df.to_parquet(view_dir / "clustering_matrix_raw.parquet")

        view_meta = {
            "view_name": view_name,
            "default_view": view_name == default_view,
            "cohort_policy": cohort_policy,
            "n_participants": int(len(scaled_view_df)),
            "n_features": int(scaled_view_df.shape[1]) if not scaled_view_df.empty else 0,
            "feature_names": view_cols,
            "include_prefixes": list(include_prefixes),
            "modalities": sorted({"environment" if prefix == "env_" else "wearable" for prefix in include_prefixes}),
            "normalization": norm_meta,
            "artifact_policy": artifact_policy,
            "created": pd.Timestamp.utcnow().isoformat(),
            "dropped_per_step": drop_log,
        }
        (view_dir / "clustering_matrix_meta.json").write_text(json.dumps(view_meta, indent=2))

        if view_name == default_view:
            default_scaled_df = scaled_view_df
            default_raw_df = raw_view_df
            default_view_meta = view_meta

    if default_scaled_df is None or default_raw_df is None or default_view_meta is None:
        raise RuntimeError(f"Default clustering view '{default_view}' was not created.")

    outcome_path = processed_path / "outcome_matrix.parquet"
    outcome_df.to_parquet(outcome_path)

    if artifact_policy["write_default_aliases"]:
        clustering_path = processed_path / "clustering_matrix.parquet"
        default_scaled_df.to_parquet(clustering_path)
        if artifact_policy["save_view_raw_matrices"]:
            raw_path = processed_path / "clustering_matrix_raw.parquet"
            default_raw_df.to_parquet(raw_path)
        clustering_meta = dict(default_view_meta)
        (processed_path / "clustering_matrix_meta.json").write_text(json.dumps(clustering_meta, indent=2))

    outcome_meta = {
        "n_participants": int(len(outcome_df)),
        "n_features": int(outcome_df.shape[1]) if not outcome_df.empty else 0,
        "modalities": ["cgm", "clinical"],
        "normalized": False,
        "created": pd.Timestamp.utcnow().isoformat(),
        "dropped_per_step": drop_log,
    }
    (processed_path / "outcome_matrix_meta.json").write_text(json.dumps(outcome_meta, indent=2))

    post_stage_counts = joined["diabetes_stage"].value_counts(dropna=False).to_dict() if "diabetes_stage" in joined.columns else {}
    balance = {}
    for stage, pre_n in pre_stage_counts.items():
        post_n = post_stage_counts.get(stage, 0)
        drop_pct = ((pre_n - post_n) / pre_n) if pre_n else None
        balance[stage] = {"pre": int(pre_n), "post": int(post_n), "drop_pct": float(drop_pct) if drop_pct is not None else None}
        if drop_pct is not None and drop_pct > 0.2:
            print(f"[WARN] Stage {stage} lost {drop_pct:.1%} of participants during assembly")
    balance_meta = {
        "stage_balance": balance,
        "created": pd.Timestamp.utcnow().isoformat(),
    }
    (processed_path / "assemble_balance.json").write_text(json.dumps(balance_meta, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Module 1 assembly")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()
    assemble(Path(args.config))
