from __future__ import annotations

from pathlib import Path
import json
import sys
import numpy as np
import matplotlib
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __name__ == "__main__" and __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    from .common import (
        load_config,
        ensure_dirs,
        load_participants,
        pull_environment_data,
    )
except ImportError:
    # Fallback when executed directly as a script
    from module1_processing.common import (  # type: ignore
        load_config,
        ensure_dirs,
        load_participants,
        pull_environment_data,
    )

ENV_PATH = "environment/environmental_sensor/leelab_anura/{pid}/{pid}_ENV.csv"


def build_environment_features(cfg_path: Path) -> None:
    cfg, base = load_config(cfg_path)
    thresholds = cfg["module1"]["qc_thresholds"]["environment"]
    summary_cfg = cfg["module1"].get("environment_feature_summaries", {})
    raw_path = Path(cfg["data"]["raw_path"].replace("${AIREADI_DATA_PATH}", str(base)))
    inter_dir = Path(cfg["data"]["intermediates_path"].replace("${AIREADI_DATA_PATH}", str(base)))
    qc_dir = Path(cfg["data"]["qc_reports_path"].replace("${AIREADI_DATA_PATH}", str(base)))

    ensure_dirs(inter_dir, qc_dir)

    participants = load_participants(raw_path)
    rows = []
    light_hourlies = []
    exclusion_reasons = {}

    def record_reason(reason: str) -> None:
        exclusion_reasons[reason] = exclusion_reasons.get(reason, 0) + 1

    valid_ranges = {
        "pm1": (thresholds.get("pm1_min"), thresholds.get("pm1_max")),
        "pm2.5": (thresholds.get("pm2_5_min"), thresholds.get("pm2_5_max")),
        "pm10": (thresholds.get("pm10_min"), thresholds.get("pm10_max")),
        "hum": (thresholds.get("humidity_min"), thresholds.get("humidity_max")),
        "temp": (thresholds.get("temperature_min_c"), thresholds.get("temperature_max_c")),
        "voc": (thresholds.get("voc_min"), thresholds.get("voc_max")),
        "nox": (thresholds.get("nox_min"), thresholds.get("nox_max")),
    }

    prop_thresholds = {
        "voc": summary_cfg.get("voc_elevation_threshold"),
        "nox": summary_cfg.get("nox_elevation_threshold"),
        "light_total": summary_cfg.get("light_total_activity_threshold"),
    }

    for pid in participants["person_id"]:
        env_path = raw_path / ENV_PATH.format(pid=pid)
        try:
            df, light_hourly = pull_environment_data(
                pid,
                env_path,
                prefix="env",
                valid_ranges=valid_ranges,
                prop_thresholds=prop_thresholds,
                strict_missing=True,
                return_hourly=True,
            )
            valid_hours = df.loc[0, "env_valid_hours"] if "env_valid_hours" in df.columns else 0
            min_hours = thresholds.get("min_valid_hours", 0)
            if valid_hours < min_hours:
                record_reason("<min_valid_hours")
                continue
            rows.append(df)
            if not light_hourly.empty:
                light_hourlies.append(light_hourly)
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
    # Drop nox median/iqr per updated feature list (keep prop_high)
    drop_cols = [c for c in features.columns if c.startswith("env_nox_") and (c.endswith("_median") or c.endswith("_iqr"))]
    features = features.drop(columns=drop_cols, errors="ignore")
    features.to_parquet(inter_dir / "environment_features.parquet")

    # Light total distribution for threshold calibration
    if light_hourlies:
        all_light = pd.concat(light_hourlies, axis=0)
        all_light = all_light.replace([np.inf, -np.inf], np.nan).dropna()
        if not all_light.empty:
            # focus resolution where most values live (0–~2), while keeping upper tail
            p99 = float(np.quantile(all_light, 0.99)) if len(all_light) else 2.0
            upper = max(2.0, p99)
            bins = 200 if upper <= 5 else 100  # finer bins in dense low range
            plt.figure(figsize=(7, 4))
            plt.hist(all_light, bins=bins, range=(0, upper), color="#3c6e71", edgecolor="white", alpha=0.9)
            plt.xlabel(f"Light total (summed channels) [0, {upper:.2f}]")
            plt.ylabel("Hourly count")
            plt.title("Light total hourly distribution (env)")
            plt.tight_layout()
            plt.savefig(qc_dir / "environment_light_total_hist.png", dpi=150)
            plt.close()

    qc = {
        "modality": "environment",
        "n_input": int(len(participants)),
        "n_passed": int(len(features)),
        "n_excluded": int(len(participants) - len(features)),
        "exclusion_reasons": exclusion_reasons,
        "thresholds_applied": thresholds,
        "prop_thresholds": prop_thresholds,
    }
    (qc_dir / "environment_qc.json").write_text(json.dumps(qc, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Module 1 environment feature extraction")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()
    build_environment_features(Path(args.config))
