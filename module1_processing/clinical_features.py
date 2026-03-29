from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

from .common import (
    load_config,
    ensure_dirs,
    load_participants,
    CLINICAL_FILE,
    CLINICAL_STAGE_COL,
    MEASUREMENT_FILE,
    HBA1C_SOURCE_VALUE,
)


def build_clinical_features(cfg_path: Path) -> None:
    cfg, base = load_config(cfg_path)
    raw_path = Path(cfg["data"]["raw_path"].replace("${AIREADI_DATA_PATH}", str(base)))
    inter_dir = Path(cfg["data"]["intermediates_path"].replace("${AIREADI_DATA_PATH}", str(base)))
    qc_dir = Path(cfg["data"]["qc_reports_path"].replace("${AIREADI_DATA_PATH}", str(base)))

    ensure_dirs(inter_dir, qc_dir)

    participants = load_participants(raw_path)

    roster_path = Path(str(CLINICAL_FILE).replace("${AIREADI_DATA_PATH}", str(base)))
    if not roster_path.exists():
        raise FileNotFoundError(f"Clinical roster not found: {roster_path}")
    roster_df = pd.read_csv(roster_path, sep="\t")

    if CLINICAL_STAGE_COL not in roster_df.columns:
        raise ValueError(f"Missing required roster column: {CLINICAL_STAGE_COL}")

    measurement_path = Path(str(MEASUREMENT_FILE).replace("${AIREADI_DATA_PATH}", str(base)))
    if not measurement_path.exists():
        raise FileNotFoundError(f"Measurement table not found: {measurement_path}")

    available_cols = pd.read_csv(measurement_path, nrows=0).columns.tolist()
    needed = {"person_id", "measurement_source_value", "value_as_number"}
    if not needed.issubset(set(available_cols)):
        missing = needed - set(available_cols)
        raise ValueError(f"Measurement table missing columns: {sorted(missing)}")

    usecols = [c for c in [
        "person_id",
        "measurement_source_value",
        "measurement_datetime",
        "measurement_date",
        "value_as_number",
    ] if c in available_cols]

    measurement = pd.read_csv(measurement_path, usecols=usecols)
    hba1c_rows = measurement[measurement["measurement_source_value"].eq(HBA1C_SOURCE_VALUE)].copy()
    if hba1c_rows.empty:
        raise ValueError("No HbA1c rows found in measurement table; update source value")

    if "measurement_datetime" in hba1c_rows.columns:
        hba1c_rows["ts"] = pd.to_datetime(hba1c_rows["measurement_datetime"], errors="coerce")
    elif "measurement_date" in hba1c_rows.columns:
        hba1c_rows["ts"] = pd.to_datetime(hba1c_rows["measurement_date"], errors="coerce")
    else:
        raise ValueError("Measurement table lacks datetime/date columns for HbA1c ordering.")

    hba1c_rows = hba1c_rows.sort_values(["person_id", "ts"]).drop_duplicates("person_id", keep="last")
    hba1c_map = hba1c_rows.set_index("person_id")["value_as_number"]

    exclusion_reasons: dict[str, int] = {}

    def record(reason: str) -> None:
        exclusion_reasons[reason] = exclusion_reasons.get(reason, 0) + 1

    rows = []
    for pid in participants["person_id"]:
        stage_val = roster_df.loc[roster_df["person_id"] == pid, CLINICAL_STAGE_COL].squeeze() if not roster_df.empty else None
        if pd.isna(stage_val):
            record("missing_stage")
            continue
        hba1c_val = hba1c_map.get(pid)
        if hba1c_val is None or pd.isna(hba1c_val):
            record("missing_hba1c")
            continue

        rows.append({
            "person_id": pid,
            "diabetes_stage": stage_val,
            "hba1c": hba1c_val,
        })

    # HbA1c strata derived from config thresholds
    strata_cfg = cfg.get("module3", {}).get("hba1c_strata_boundaries", {})
    well = strata_cfg.get("well_controlled")
    moderate = strata_cfg.get("moderate")
    if well is None or moderate is None:
        raise ValueError("module3.hba1c_strata_boundaries must define well_controlled and moderate cutpoints")

    def stratify(val):
        if pd.isna(val):
            return None
        if val < well:
            return "well_controlled"
        if val < moderate:
            return "moderate"
        return "poor"

    features_df = pd.DataFrame(rows)
    if features_df.empty:
        raise ValueError("Clinical features empty after exclusions; check stage/hba1c availability")

    features_df["hba1c_stratum"] = features_df["hba1c"].apply(stratify)

    features_df = features_df.set_index("person_id")
    features_df.to_parquet(inter_dir / "clinical_features.parquet")

    qc = {
        "modality": "clinical",
        "n_input": int(len(participants)),
        "n_passed": int(len(features_df)),
        "n_excluded": int(len(participants) - len(features_df)),
        "exclusion_reasons": exclusion_reasons,
        "thresholds_applied": {},
        "notes": "Clinical features include hba1c, stratum, diabetes_stage",
    }
    (qc_dir / "clinical_qc.json").write_text(json.dumps(qc, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Module 1 clinical feature extraction")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()
    build_clinical_features(Path(args.config))
