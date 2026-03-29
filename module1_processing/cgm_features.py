from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

from .common import (
    load_config,
    ensure_dirs,
    load_participants,
    json_to_df,
    count_unique_days,
)


CGM_PATH_TEMPLATE = "wearable_blood_glucose/continuous_glucose_monitoring/dexcom_g6/{pid}/{pid}_DEX.json"


def build_cgm_features(cfg_path: Path) -> None:
    cfg, base = load_config(cfg_path)
    thresholds = cfg["module1"]["qc_thresholds"]["cgm"]
    raw_path = Path(cfg["data"]["raw_path"].replace("${AIREADI_DATA_PATH}", str(base)))
    inter_dir = Path(cfg["data"]["intermediates_path"].replace("${AIREADI_DATA_PATH}", str(base)))
    qc_dir = Path(cfg["data"]["qc_reports_path"].replace("${AIREADI_DATA_PATH}", str(base)))

    ensure_dirs(inter_dir, qc_dir)

    participants = load_participants(raw_path)
    rows = []
    exclusion_reasons = {}

    for pid in participants["person_id"]:
        path = raw_path / CGM_PATH_TEMPLATE.format(pid=pid)
        df = json_to_df(path, strict_missing=True)

        ts_start = pd.to_datetime(df["effective_time_frame.time_interval.start_date_time"], errors="coerce", utc=True)
        ts_end = pd.to_datetime(df["effective_time_frame.time_interval.end_date_time"], errors="coerce", utc=True)
        values = pd.to_numeric(df["blood_glucose.value"], errors="coerce")
        mid_ts = ts_start + (ts_end - ts_start) / 2

        glucose = pd.DataFrame({"ts": mid_ts, "value": values}).dropna()
        glucose = glucose[(glucose["value"] >= thresholds.get("glucose_min_mg_dl")) & (glucose["value"] <= thresholds.get("glucose_max_mg_dl"))]

        n_days = count_unique_days(glucose["ts"]) if not glucose.empty else 0
        if n_days < thresholds.get("min_wear_days", 0):
            exclusion_reasons["<min_wear_days"] = exclusion_reasons.get("<min_wear_days", 0) + 1
            continue

        mean_glu = glucose["value"].mean()
        std_glu = glucose["value"].std(ddof=0)
        glycemic_cv = float(std_glu / mean_glu * 100) if mean_glu and not pd.isna(mean_glu) else None
        tir = float(((glucose["value"] >= 70) & (glucose["value"] <= 180)).mean()) if not glucose.empty else None

        rows.append(pd.DataFrame({
            "person_id": [pid],
            "glycemic_cv": [glycemic_cv],
            "mean_glucose": [mean_glu],
            "time_in_range": [tir],
        }))

    features = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["person_id", "glycemic_cv", "mean_glucose", "time_in_range"])
    if features.empty:
        raise ValueError("CGM features empty after applying QC; check inputs and thresholds")
    features = features.set_index("person_id")

    features.to_parquet(inter_dir / "cgm_features.parquet")

    qc = {
        "modality": "cgm",
        "n_input": int(len(participants)),
        "n_passed": int(len(features)),
        "n_excluded": int(len(participants) - len(features)),
        "exclusion_reasons": exclusion_reasons,
        "thresholds_applied": thresholds,
    }
    (qc_dir / "cgm_qc.json").write_text(json.dumps(qc, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Module 1 CGM feature extraction")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()
    build_cgm_features(Path(args.config))
