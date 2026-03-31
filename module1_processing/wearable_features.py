from __future__ import annotations

from pathlib import Path
import json
import sys
import pandas as pd
import numpy as np

if __name__ == "__main__" and __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    from .common import (
        load_config,
        ensure_dirs,
        load_participants,
        json_to_df,
        pull_monitor_data,
        pull_sleep_data,
        wear_minutes_from_heart_rate,
    )
except ImportError:
    # Fallback when executed as a script (python module1_processing/wearable_features.py ...)
    from module1_processing.common import (  # type: ignore
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


def _prop_below_threshold(df: pd.DataFrame, threshold: float, value_col: str, time_col: str, wear_minutes: pd.DatetimeIndex | None,
                          min_value: float | None = None, max_value: float | None = None) -> float | None:
    """Compute proportion of values below threshold after basic cleaning and wear filtering."""
    if df is None or df.empty or threshold is None:
        return None
    ts = pd.to_datetime(df[time_col], errors="coerce", utc=True) if time_col in df.columns else pd.Series(dtype="datetime64[ns, UTC]")
    v = pd.to_numeric(df[value_col], errors="coerce") if value_col in df.columns else pd.Series(dtype=float)
    if wear_minutes is not None and len(wear_minutes) and not ts.empty:
        keep = ts.dt.floor("min").isin(wear_minutes)
        ts = ts.loc[keep]
        v = v.loc[keep]
    if min_value is not None:
        v = v.mask(v < min_value)
    if max_value is not None:
        v = v.mask(v > max_value)
    v = v.dropna()
    if v.empty:
        return None
    return float((v < threshold).mean())


def _resting_hr_sleep_median(
    hr_df: pd.DataFrame,
    sleep_df: pd.DataFrame,
    hr_col: str,
    hr_min: float,
    hr_max: float,
    hr_time_col: str,
    sleep_start_col: str,
    sleep_end_col: str,
) -> float | None:
    """Median heart rate during sleep intervals; None if no overlap."""
    if hr_df is None or sleep_df is None or hr_df.empty or sleep_df.empty:
        return None
    if hr_time_col not in hr_df.columns or hr_col not in hr_df.columns:
        return None
    if sleep_start_col not in sleep_df.columns or sleep_end_col not in sleep_df.columns:
        return None

    hr_ts = pd.to_datetime(hr_df[hr_time_col], errors="coerce", utc=True)
    hr_vals = pd.to_numeric(hr_df[hr_col], errors="coerce")
    keep_hr = hr_vals.between(hr_min, hr_max) & hr_ts.notna()
    hr_ts = hr_ts[keep_hr]
    hr_vals = hr_vals[keep_hr]
    if hr_ts.empty:
        return None

    sleep_start = pd.to_datetime(sleep_df[sleep_start_col], errors="coerce", utc=True)
    sleep_end = pd.to_datetime(sleep_df[sleep_end_col], errors="coerce", utc=True)
    mask = np.zeros(len(hr_ts), dtype=bool)
    for s, e in zip(sleep_start, sleep_end):
        if pd.isna(s) or pd.isna(e):
            continue
        mask |= hr_ts.between(s, e, inclusive="both").to_numpy()

    vals = hr_vals[mask]
    vals = pd.to_numeric(vals, errors="coerce").dropna()
    if vals.empty:
        return None
    return float(vals.median())


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

            # Raw SpO2 for proportion below 95%
            o2_prop_low = None
            try:
                o2_raw = json_to_df(raw_path / O2_PATH.format(pid=pid), strict_missing=True)
                o2_prop_low = _prop_below_threshold(
                    o2_raw,
                    threshold=95,
                    value_col="oxygen_saturation.value",
                    time_col="effective_time_frame.date_time",
                    wear_minutes=wear_mins,
                    min_value=thresholds.get("oxygen_saturation_min"),
                    max_value=thresholds.get("oxygen_saturation_max"),
                )
            except FileNotFoundError:
                record_reason("missing_file_oxygen_sat")
            except ValueError:
                record_reason("value_error_oxygen_sat")
            except Exception:
                record_reason("other_error_oxygen_sat")

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

            resting_hr = None
            try:
                sleep_path = raw_path / SLEEP_PATH.format(pid=pid)
                sleep_raw = json_to_df(sleep_path, strict_missing=True)
                resting_hr = _resting_hr_sleep_median(
                    hr_df=hr_df,
                    sleep_df=sleep_raw,
                    hr_col="heart_rate.value",
                    hr_min=thresholds.get("heart_rate_min", 25),
                    hr_max=thresholds.get("heart_rate_max", 250),
                    hr_time_col="effective_time_frame.date_time",
                    sleep_start_col="effective_time_frame.time_interval.start_date_time",
                    sleep_end_col="effective_time_frame.time_interval.end_date_time",
                )
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
            merged["heart_rate_resting_median"] = resting_hr
            merged["oxygen_sat_prop_below_95"] = o2_prop_low
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
