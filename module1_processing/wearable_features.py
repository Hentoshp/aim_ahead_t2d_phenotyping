from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

from .common import (
    load_config,
    ensure_dirs,
    load_participants,
    json_to_df,
    pull_monitor_data,
    pull_sleep_data,
    wear_minutes_from_heart_rate,
)

HR_PATH = "wearable_activity_monitor/heart_rate/garmin_vivosmart5/{pid}/{pid}_heartrate.json"
O2_PATH = "wearable_activity_monitor/oxygen_saturation/garmin_vivosmart5/{pid}/{pid}_oxygensaturation.json"
ACTIVITY_PATH = "wearable_activity_monitor/physical_activity/garmin_vivosmart5/{pid}/{pid}_activity.json"
CALORIES_PATH = "wearable_activity_monitor/physical_activity_calorie/garmin_vivosmart5/{pid}/{pid}_calorie.json"
RESP_PATH = "wearable_activity_monitor/respiratory_rate/garmin_vivosmart5/{pid}/{pid}_respiratoryrate.json"
STRESS_PATH = "wearable_activity_monitor/stress/garmin_vivosmart5/{pid}/{pid}_stress.json"
SLEEP_PATH = "wearable_activity_monitor/sleep/garmin_vivosmart5/{pid}/{pid}_sleep.json"


def build_wearable_features(cfg_path: Path) -> None:
    cfg, base = load_config(cfg_path)
    thresholds = cfg["module1"]["qc_thresholds"]["wearable"]
    raw_path = Path(cfg["data"]["raw_path"].replace("${AIREADI_DATA_PATH}", str(base)))
    inter_dir = Path(cfg["data"]["intermediates_path"].replace("${AIREADI_DATA_PATH}", str(base)))
    qc_dir = Path(cfg["data"]["qc_reports_path"].replace("${AIREADI_DATA_PATH}", str(base)))

    ensure_dirs(inter_dir, qc_dir)

    participants = load_participants(raw_path)
    rows = []
    exclusion_reasons = {}
    exclusion_records = []

    def record_reason(reason: str) -> None:
        exclusion_reasons[reason] = exclusion_reasons.get(reason, 0) + 1
        exclusion_records.append({"person_id": pid, "reason": reason})

    for pid in participants["person_id"]:
        try:
            hr_path = raw_path / HR_PATH.format(pid=pid)
            hr_df = json_to_df(hr_path, strict_missing=True)
            wear_mins = wear_minutes_from_heart_rate(
                hr_df,
                hr_min=thresholds.get("heart_rate_min", 25),
                hr_max=thresholds.get("heart_rate_max", 250),
            )

            hr_features = pull_monitor_data(
                pid,
                hr_path,
                value_col="heart_rate.value",
                prefix="heart_rate",
                time_col="effective_time_frame.date_time",
                wear_minutes=wear_mins,
                min_samples_per_hour=10,
                min_value=thresholds.get("heart_rate_min"),
                max_value=thresholds.get("heart_rate_max"),
                strict_missing=True,
            )

            coverage = 0.0
            ndays = hr_features.loc[0, "heart_rate_ndays"] or 0
            valid_hours = hr_features.loc[0, "heart_rate_valid_hours"] or 0
            if ndays:
                coverage = valid_hours / (24 * ndays)
            if coverage < thresholds.get("min_heart_rate_valid_hour_coverage", 0):
                record_reason("<min_hr_coverage>")
                continue

            # Downstream modalities: on failure, record reason but keep participant with NaNs
            def safe_monitor(path_tmpl, prefix, value_col=None, time_col=None, start_col=None, end_col=None, daily_agg=None,
                             min_value=None, max_value=None):
                path = raw_path / path_tmpl.format(pid=pid)
                try:
                    return pull_monitor_data(
                        pid,
                        path,
                        value_col=value_col,
                        prefix=prefix,
                        time_col=time_col,
                        start_col=start_col,
                        end_col=end_col,
                        wear_minutes=wear_mins,
                        daily_agg=daily_agg,
                        min_value=min_value,
                        max_value=max_value,
                        strict_missing=True,
                    )
                except FileNotFoundError:
                    record_reason(f"missing_file_{prefix}")
                except ValueError:
                    record_reason(f"value_error_{prefix}")
                except Exception:
                    record_reason(f"other_error_{prefix}")
                # fallback empty row with expected columns
                return pull_monitor_data(
                    pid,
                    path,
                    value_col=value_col,
                    prefix=prefix,
                    time_col=time_col,
                    start_col=start_col,
                    end_col=end_col,
                    wear_minutes=None,
                    daily_agg=daily_agg,
                    min_value=min_value,
                    max_value=max_value,
                    strict_missing=False,
                )

            o2_features = safe_monitor(
                O2_PATH, "oxygen_sat", value_col="oxygen_saturation.value",
                time_col="effective_time_frame.date_time",
                min_value=thresholds.get("oxygen_saturation_min"),
                max_value=thresholds.get("oxygen_saturation_max"),
            )

            activity_features = safe_monitor(
                ACTIVITY_PATH, "physical_activity", value_col="base_movement_quantity.value",
                start_col="effective_time_frame.time_interval.start_date_time",
                end_col="effective_time_frame.time_interval.end_date_time",
                daily_agg="sum",
                min_value=thresholds.get("physical_activity_min"),
            )

            calorie_features = safe_monitor(
                CALORIES_PATH, "calories", value_col="calories_value.value",
                time_col="effective_time_frame.date_time",
                daily_agg="sum",
                min_value=thresholds.get("calories_min"),
            )

            resp_features = safe_monitor(
                RESP_PATH, "respiratory_rate", value_col="respiratory_rate.value",
                time_col="effective_time_frame.date_time",
                min_value=thresholds.get("respiratory_rate_min"),
                max_value=thresholds.get("respiratory_rate_max"),
            )

            stress_features = safe_monitor(
                STRESS_PATH, "stress", value_col="stress.value",
                time_col="effective_time_frame.date_time",
                min_value=thresholds.get("stress_min"),
                max_value=thresholds.get("stress_max"),
            )

            try:
                sleep_path = raw_path / SLEEP_PATH.format(pid=pid)
                sleep_features = pull_sleep_data(
                    pid,
                    sleep_path,
                    prefix="sleep",
                    stage_col="sleep_stage_state",
                    start_col="effective_time_frame.time_interval.start_date_time",
                    end_col="effective_time_frame.time_interval.end_date_time",
                    wear_minutes=wear_mins,
                    strict_missing=True,
                )
            except FileNotFoundError:
                record_reason("missing_file_sleep")
                sleep_features = pull_sleep_data(
                    pid,
                    sleep_path,
                    prefix="sleep",
                    stage_col="sleep_stage_state",
                    start_col="effective_time_frame.time_interval.start_date_time",
                    end_col="effective_time_frame.time_interval.end_date_time",
                    wear_minutes=None,
                    strict_missing=False,
                )
            except ValueError:
                record_reason("value_error_sleep")
                sleep_features = pull_sleep_data(
                    pid,
                    sleep_path,
                    prefix="sleep",
                    stage_col="sleep_stage_state",
                    start_col="effective_time_frame.time_interval.start_date_time",
                    end_col="effective_time_frame.time_interval.end_date_time",
                    wear_minutes=None,
                    strict_missing=False,
                )
            except Exception:
                record_reason("other_error_sleep")
                sleep_features = pull_sleep_data(
                    pid,
                    sleep_path,
                    prefix="sleep",
                    stage_col="sleep_stage_state",
                    start_col="effective_time_frame.time_interval.start_date_time",
                    end_col="effective_time_frame.time_interval.end_date_time",
                    wear_minutes=None,
                    strict_missing=False,
                )

            merged = hr_features.merge(o2_features, on="person_id", how="left")
            merged = merged.merge(activity_features, on="person_id", how="left")
            merged = merged.merge(calorie_features, on="person_id", how="left")
            merged = merged.merge(resp_features, on="person_id", how="left")
            merged = merged.merge(stress_features, on="person_id", how="left")
            merged = merged.merge(sleep_features, on="person_id", how="left")
            rows.append(merged)

        except FileNotFoundError as e:
            record_reason("missing_file")
        except ValueError as e:
            record_reason("value_error")
        except Exception as e:
            record_reason("other_error")

    features = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["person_id"])
    if features.empty:
        raise ValueError("Wearable features empty after processing; check input data and thresholds")

    features = features.set_index("person_id")
    features.to_parquet(inter_dir / "wearable_features.parquet")

    if exclusion_records:
        import csv
        excl_path = qc_dir / "wearable_exclusions.csv"
        with excl_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["person_id", "reason"])
            writer.writeheader()
            writer.writerows(exclusion_records)

    qc = {
        "modality": "wearable",
        "n_input": int(len(participants)),
        "n_passed": int(len(features)),
        "n_excluded": int(len(participants) - len(features)),
        "exclusion_reasons": exclusion_reasons,
        "thresholds_applied": thresholds,
    }
    (qc_dir / "wearable_qc.json").write_text(json.dumps(qc, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Module 1 wearable feature extraction")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()
    build_wearable_features(Path(args.config))
