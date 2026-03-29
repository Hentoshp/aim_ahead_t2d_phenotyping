from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

from .common import (
    load_config,
    ensure_dirs,
    load_participants,
    pull_environment_data,
)

ENV_PATH = "environment/environmental_sensor/leelab_anura/{pid}/{pid}_ENV.csv"


def build_environment_features(cfg_path: Path) -> None:
    cfg, base = load_config(cfg_path)
    thresholds = cfg["module1"]["qc_thresholds"]["environment"]
    raw_path = Path(cfg["data"]["raw_path"].replace("${AIREADI_DATA_PATH}", str(base)))
    inter_dir = Path(cfg["data"]["intermediates_path"].replace("${AIREADI_DATA_PATH}", str(base)))
    qc_dir = Path(cfg["data"]["qc_reports_path"].replace("${AIREADI_DATA_PATH}", str(base)))

    ensure_dirs(inter_dir, qc_dir)

    participants = load_participants(raw_path)
    rows = []
    exclusion_reasons = {}

    def record_reason(reason: str) -> None:
        exclusion_reasons[reason] = exclusion_reasons.get(reason, 0) + 1

    valid_ranges = {
        "pm1": (thresholds.get("pm1_min"), thresholds.get("pm1_max")),
        "pm2.5": (thresholds.get("pm2_5_min"), thresholds.get("pm2_5_max")),
        "pm4": (thresholds.get("pm4_min"), thresholds.get("pm4_max")),
        "pm10": (thresholds.get("pm10_min"), thresholds.get("pm10_max")),
        "hum": (thresholds.get("humidity_min"), thresholds.get("humidity_max")),
        "temp": (thresholds.get("temperature_min_c"), thresholds.get("temperature_max_c")),
        "voc": (thresholds.get("voc_min"), thresholds.get("voc_max")),
        "nox": (thresholds.get("nox_min"), thresholds.get("nox_max")),
        "lch0": (thresholds.get("light_min"), thresholds.get("light_max")),
        "lch1": (thresholds.get("light_min"), thresholds.get("light_max")),
        "lch2": (thresholds.get("light_min"), thresholds.get("light_max")),
        "lch3": (thresholds.get("light_min"), thresholds.get("light_max")),
        "lch6": (thresholds.get("light_min"), thresholds.get("light_max")),
        "lch7": (thresholds.get("light_min"), thresholds.get("light_max")),
        "lch8": (thresholds.get("light_min"), thresholds.get("light_max")),
        "lch9": (thresholds.get("light_min"), thresholds.get("light_max")),
        "lch10": (thresholds.get("light_min"), thresholds.get("light_max")),
        "lch11": (thresholds.get("light_min"), thresholds.get("light_max")),
    }

    for pid in participants["person_id"]:
        env_path = raw_path / ENV_PATH.format(pid=pid)
        try:
            df = pull_environment_data(pid, env_path, prefix="env", valid_ranges=valid_ranges, strict_missing=True)
            valid_hours = df.loc[0, "env_valid_hours"] if "env_valid_hours" in df.columns else 0
            min_hours = thresholds.get("min_valid_hours", 0)
            if valid_hours < min_hours:
                record_reason("<min_valid_hours")
                continue
            rows.append(df)
        except FileNotFoundError:
            record_reason("missing_file")
        except ValueError:
            record_reason("value_error")
        except Exception:
            record_reason("other_error")

    features = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["person_id"])
    if features.empty:
        raise ValueError("Environment features empty; check input files and thresholds")

    features = features.set_index("person_id")
    features.to_parquet(inter_dir / "environment_features.parquet")

    qc = {
        "modality": "environment",
        "n_input": int(len(participants)),
        "n_passed": int(len(features)),
        "n_excluded": int(len(participants) - len(features)),
        "exclusion_reasons": exclusion_reasons,
        "thresholds_applied": thresholds,
    }
    (qc_dir / "environment_qc.json").write_text(json.dumps(qc, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Module 1 environment feature extraction")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()
    build_environment_features(Path(args.config))
