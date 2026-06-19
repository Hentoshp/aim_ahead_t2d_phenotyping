from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from .common import load_config, ensure_dirs
from parquet_utils import read_parquet_with_compat_hint

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
OUTCOME_COLUMNS = [
    "glycemic_cv",
    "mean_glucose",
    "time_in_range",
    "time_below_70",
    "time_below_54",
    "hba1c",
    "diabetes_stage",
]


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


def _infer_clustering_modalities(include_prefixes: tuple[str, ...]) -> tuple[str, ...]:
    modalities = []
    if any(prefix == "env_" for prefix in include_prefixes):
        modalities.append("environment")
    if any(prefix != "env_" for prefix in include_prefixes):
        modalities.append("wearable")
    return tuple(modalities)


def _resolve_required_modalities(
    include_prefixes: tuple[str, ...],
    cohort_policy: str,
    configured_modalities: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    if configured_modalities:
        invalid = sorted(set(configured_modalities) - set(MODALITIES))
        if invalid:
            raise ValueError(f"Unsupported required_modalities values: {invalid}")
        return tuple(modality for modality in MODALITIES if modality in configured_modalities)

    if cohort_policy == "common":
        return tuple(MODALITIES)
    if cohort_policy == "view_specific":
        required = set(_infer_clustering_modalities(include_prefixes)) | {"cgm", "clinical"}
        return tuple(modality for modality in MODALITIES if modality in required)
    raise ValueError(f"Unknown cohort_policy: {cohort_policy}")


def _load_clustering_views(cfg: dict) -> tuple[str, str, dict[str, dict[str, tuple[str, ...]]]]:
    view_cfg = cfg.get("module1", {}).get("clustering_views", {})
    cohort_policy = str(view_cfg.get("cohort_policy", "view_specific"))
    if cohort_policy not in {"common", "view_specific"}:
        raise ValueError("module1.clustering_views.cohort_policy must be 'common' or 'view_specific'.")

    default_views = {
        "wearable": WEARABLE_CLUSTERING_PREFIXES,
        "environment": ENVIRONMENT_CLUSTERING_PREFIXES,
        "wearable_environment": CLUSTERING_PREFIXES,
    }
    raw_views = view_cfg.get("views") or {
        name: {"include_prefixes": list(prefixes)}
        for name, prefixes in default_views.items()
    }

    views: dict[str, dict[str, tuple[str, ...]]] = {}
    for view_name, spec in raw_views.items():
        prefixes = tuple(spec.get("include_prefixes", []))
        if not prefixes:
            raise ValueError(f"Clustering view '{view_name}' must define include_prefixes.")
        configured_required = tuple(spec.get("required_modalities", []))
        views[str(view_name)] = {
            "include_prefixes": prefixes,
            "required_modalities": _resolve_required_modalities(
                include_prefixes=prefixes,
                cohort_policy=cohort_policy,
                configured_modalities=configured_required if configured_required else None,
            ),
        }

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


def _join_modalities(
    modality_frames: dict[str, pd.DataFrame],
    required_modalities: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, int]]:
    if not required_modalities:
        raise ValueError("required_modalities cannot be empty.")

    joined = modality_frames[required_modalities[0]].copy()
    drop_log: dict[str, int] = {}
    for modality in required_modalities[1:]:
        df = modality_frames[modality]
        before = len(joined)
        joined = joined.join(df, how="inner")
        drop_log[modality] = int(before - len(joined))
    return joined, drop_log


def _apply_missing_strategy(
    clustering_df: pd.DataFrame,
    outcome_df: pd.DataFrame,
    missing_strategy: str,
    drop_log: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    updated_log = dict(drop_log)
    if missing_strategy == "drop":
        before = len(clustering_df)
        mask = clustering_df.notna().all(axis=1)
        clustering_df = clustering_df.loc[mask]
        outcome_df = outcome_df.loc[mask]
        updated_log["missing_values"] = int(before - len(clustering_df))
    elif missing_strategy == "impute":
        raise NotImplementedError("missing_strategy 'impute' not implemented yet; use 'drop' or 'none'")
    elif missing_strategy == "none":
        pass
    else:
        raise ValueError(f"Unknown missing_strategy: {missing_strategy}")
    return clustering_df, outcome_df, updated_log


def _stage_balance(pre_stage_counts: dict, post_stage_counts: dict) -> dict:
    balance = {}
    for stage, pre_n in pre_stage_counts.items():
        post_n = post_stage_counts.get(stage, 0)
        drop_pct = ((pre_n - post_n) / pre_n) if pre_n else None
        balance[stage] = {
            "pre": int(pre_n),
            "post": int(post_n),
            "drop_pct": float(drop_pct) if drop_pct is not None else None,
        }
    return {
        "stage_balance": balance,
        "created": pd.Timestamp.utcnow().isoformat(),
    }


def _emit_stage_balance_warnings(balance_meta: dict, prefix: str = "") -> None:
    for stage, stats in balance_meta.get("stage_balance", {}).items():
        drop_pct = stats.get("drop_pct")
        if drop_pct is not None and drop_pct > 0.2:
            stage_prefix = f"{prefix} " if prefix else ""
            print(f"[WARN] {stage_prefix}Stage {stage} lost {drop_pct:.1%} of participants during assembly")


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

    clinical_df_full = read_parquet_with_compat_hint(inter_dir / "clinical_features.parquet")
    pre_stage_counts = clinical_df_full["diabetes_stage"].value_counts(dropna=False).to_dict()

    modality_frames: dict[str, pd.DataFrame] = {}
    for modality in MODALITIES:
        path = inter_dir / f"{modality}_features.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing intermediate: {path}")
        modality_frames[modality] = read_parquet_with_compat_hint(path)

    cohort_policy, default_view, clustering_views = _load_clustering_views(cfg)
    skew_prefixes = (
        "calories_",
        "respiratory_rate_",
        "env_pm1_",
        "env_pm2.5_",
        "env_pm10_",
        "env_light_total_",
    )
    skew_contains = ("sleep_total",)
    missing_strategy = cfg["module1"].get("missing_strategy", "drop")

    default_scaled_df: pd.DataFrame | None = None
    default_raw_df: pd.DataFrame | None = None
    default_view_meta: dict | None = None
    default_outcome_df: pd.DataFrame | None = None
    default_outcome_meta: dict | None = None
    default_balance_meta: dict | None = None

    for view_name, view_spec in clustering_views.items():
        include_prefixes = view_spec["include_prefixes"]
        required_modalities = view_spec["required_modalities"]
        joined, drop_log = _join_modalities(modality_frames, required_modalities)
        if joined.empty:
            raise ValueError(
                f"Assembly produced empty feature matrix for view '{view_name}'; "
                "check upstream modality outputs and QC thresholds."
            )

        clustering_cols = [c for c in joined.columns if _is_clustering_feature(c)]
        outcome_cols = [c for c in OUTCOME_COLUMNS if c in joined.columns]
        clustering_df = joined[clustering_cols].copy()
        outcome_df = joined[outcome_cols].copy()

        skew_cols = [
            c for c in clustering_df.columns
            if (c.startswith(skew_prefixes) or any(s in c for s in skew_contains))
            and not c.endswith("_prop_high")
        ]
        for col in skew_cols:
            clustering_df[col] = np.log1p(clustering_df[col].clip(lower=0))

        clustering_df, outcome_df, drop_log = _apply_missing_strategy(
            clustering_df=clustering_df,
            outcome_df=outcome_df,
            missing_strategy=missing_strategy,
            drop_log=drop_log,
        )

        clustering_df_raw = clustering_df.copy()
        if cfg["module1"].get("normalization") == "standard_scaler" and not clustering_df.empty:
            scaler = StandardScaler()
            scaled = scaler.fit_transform(clustering_df.astype(float))
            clustering_df = pd.DataFrame(scaled, index=clustering_df.index, columns=clustering_df.columns)
            norm_meta = {"method": "StandardScaler", "with_mean": True, "with_std": True}
        else:
            norm_meta = {"method": None}

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

        outcome_df.to_parquet(view_dir / "outcome_matrix.parquet")

        view_meta = {
            "view_name": view_name,
            "default_view": view_name == default_view,
            "cohort_policy": cohort_policy,
            "required_modalities": list(required_modalities),
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

        outcome_meta = {
            "view_name": view_name,
            "default_view": view_name == default_view,
            "cohort_policy": cohort_policy,
            "required_modalities": list(required_modalities),
            "n_participants": int(len(outcome_df)),
            "n_features": int(outcome_df.shape[1]) if not outcome_df.empty else 0,
            "modalities": ["cgm", "clinical"],
            "normalized": False,
            "created": pd.Timestamp.utcnow().isoformat(),
            "dropped_per_step": drop_log,
        }
        (view_dir / "outcome_matrix_meta.json").write_text(json.dumps(outcome_meta, indent=2))

        post_stage_counts = outcome_df["diabetes_stage"].value_counts(dropna=False).to_dict() if "diabetes_stage" in outcome_df.columns else {}
        balance_meta = _stage_balance(pre_stage_counts, post_stage_counts)
        (view_dir / "assemble_balance.json").write_text(json.dumps(balance_meta, indent=2))

        if view_name == default_view:
            default_scaled_df = scaled_view_df
            default_raw_df = raw_view_df
            default_view_meta = view_meta
            default_outcome_df = outcome_df.copy()
            default_outcome_meta = outcome_meta
            default_balance_meta = balance_meta
            if artifact_policy["save_common_raw_matrix"]:
                common_raw_path = processed_path / "clustering_matrix_common_raw.parquet"
                default_raw_df.to_parquet(common_raw_path)

    if (
        default_scaled_df is None
        or default_raw_df is None
        or default_view_meta is None
        or default_outcome_df is None
        or default_outcome_meta is None
        or default_balance_meta is None
    ):
        raise RuntimeError(f"Default clustering view '{default_view}' was not created.")

    outcome_path = processed_path / "outcome_matrix.parquet"
    default_outcome_df.to_parquet(outcome_path)

    if artifact_policy["write_default_aliases"]:
        clustering_path = processed_path / "clustering_matrix.parquet"
        default_scaled_df.to_parquet(clustering_path)
        if artifact_policy["save_view_raw_matrices"]:
            raw_path = processed_path / "clustering_matrix_raw.parquet"
            default_raw_df.to_parquet(raw_path)
        clustering_meta = dict(default_view_meta)
        (processed_path / "clustering_matrix_meta.json").write_text(json.dumps(clustering_meta, indent=2))

    (processed_path / "outcome_matrix_meta.json").write_text(json.dumps(default_outcome_meta, indent=2))
    (processed_path / "assemble_balance.json").write_text(json.dumps(default_balance_meta, indent=2))
    _emit_stage_balance_warnings(default_balance_meta)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Module 1 assembly")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()
    assemble(Path(args.config))
