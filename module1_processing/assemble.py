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

CLUSTERING_PREFIXES = (
    "heart_rate_",
    "oxygen_sat_",
    "physical_activity_",
    "calories_",
    "respiratory_rate_",
    "stress_",
    "sleep_",
    "env_",
)

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


def assemble(cfg_path: Path) -> None:
    cfg, base = load_config(cfg_path)
    inter_dir = Path(cfg["data"]["intermediates_path"].replace("${AIREADI_DATA_PATH}", str(base)))
    processed_path = Path(cfg["data"]["processed_path"].replace("${AIREADI_DATA_PATH}", str(base)))

    ensure_dirs(processed_path)

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

    clustering_cols = [c for c in joined.columns if _is_clustering_feature(c)]
    outcome_cols = [c for c in [
        "glycemic_cv", "mean_glucose", "time_in_range",
        "hba1c", "hba1c_stratum", "diabetes_stage"
    ] if c in joined.columns]

    clustering_df = joined[clustering_cols]
    outcome_df = joined[outcome_cols]

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

    # Save unscaled, transformed snapshot for diagnostics
    raw_path = processed_path / "clustering_matrix_raw.parquet"
    clustering_df.to_parquet(raw_path)

    if cfg["module1"].get("normalization") == "standard_scaler" and not clustering_df.empty:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(clustering_df.astype(float))
        clustering_df = pd.DataFrame(scaled, index=clustering_df.index, columns=clustering_df.columns)
        norm_meta = {"method": "StandardScaler", "with_mean": True, "with_std": True}
    else:
        norm_meta = {"method": None}

    clustering_path = processed_path / "clustering_matrix.parquet"
    clustering_df.to_parquet(clustering_path)
    outcome_path = processed_path / "outcome_matrix.parquet"
    outcome_df.to_parquet(outcome_path)

    clustering_meta = {
        "n_participants": int(len(clustering_df)),
        "n_features": int(clustering_df.shape[1]) if not clustering_df.empty else 0,
        "modalities": ["wearable", "environment"],
        "normalization": norm_meta,
        "created": pd.Timestamp.utcnow().isoformat(),
        "dropped_per_step": drop_log,
    }
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
