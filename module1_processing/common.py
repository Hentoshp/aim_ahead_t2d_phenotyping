"""Shared utilities for Module 1 processing.

- Loads config and resolves paths using AIREADI_DATA_PATH from .env
- Provides helpers for wearable/environment feature extraction
- Stays import-safe (no top-level I/O)
"""
from __future__ import annotations

import os
from pathlib import Path
import json
import pandas as pd
import numpy as np
import yaml
from dotenv import load_dotenv

# Clinical roster/measurement defaults (override if your schema differs)
CLINICAL_FILE = Path("${AIREADI_DATA_PATH}/raw/participants.tsv")
CLINICAL_STAGE_COL = "study_group"

# OMOP measurement extraction (long format) — source-value filter
MEASUREMENT_FILE = Path("${AIREADI_DATA_PATH}/raw/clinical_data/measurement.csv")
HBA1C_SOURCE_VALUE = "import_hba1c, Hemoglobin A1c/Hemoglobin.total in "  # measurement_source_value for HbA1c

# Environment variables expected in config
ENV_VALUE_COLS = [
    "pm1", "pm2.5", "pm4", "pm10",
    "hum", "temp", "voc", "nox",
    "lch0", "lch1", "lch2", "lch3", "lch6", "lch7", "lch8", "lch9", "lch10", "lch11",
]

SLEEP_INTERVAL_HOURS_MAX = 24


def load_config(cfg_path: Path) -> tuple[dict, Path]:
    """Load YAML config and resolve AIREADI_DATA_PATH from .env."""
    load_dotenv()
    data_root = os.getenv("AIREADI_DATA_PATH")
    if not data_root:
        raise EnvironmentError("AIREADI_DATA_PATH not set; define it in .env")

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    base = Path(data_root).expanduser()
    return cfg, base


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def load_participants(raw_path: Path, cohort_flags=None) -> pd.DataFrame:
    """Load participant roster and filter by required cohort flags. Fail fast if missing."""
    roster_path = raw_path / "participants.tsv"
    if not roster_path.exists():
        raise FileNotFoundError(f"Participant roster not found: {roster_path}")

    cohort_flags = cohort_flags or [
        "clinical_data",
        "environment",
        "wearable_activity_monitor",
        "wearable_blood_glucose",
    ]

    participants = pd.read_csv(roster_path, sep="\t")
    missing_flags = [c for c in cohort_flags if c not in participants.columns]
    if missing_flags:
        raise ValueError(f"Roster missing required cohort flags: {missing_flags}")

    filtered = participants[participants[cohort_flags].all(axis=1)] if cohort_flags else participants.copy()

    dropped_columns = cohort_flags + [
        "clinical_site",
        "study_visit_date",
        "cardiac_ecg",
        "retinal_flio",
        "retinal_oct",
        "retinal_octa",
        "retinal_photography",
    ]
    drop_cols_present = [c for c in dropped_columns if c in filtered.columns]
    filtered = filtered.drop(columns=drop_cols_present).reset_index(drop=True)
    return filtered


def json_to_df(path: Path, key=None, strict_missing: bool = False) -> pd.DataFrame:
    """Process JSON file into a dataframe. Optionally fail fast on missing."""
    try:
        with open(path) as f:
            j = json.load(f)
    except FileNotFoundError:
        if strict_missing:
            raise
        return pd.DataFrame()

    body = j.get("body", {})
    if key is None:
        # first list-valued key
        key = next((k for k, v in body.items() if isinstance(v, list)), None)
    if key is None or key not in body:
        return pd.DataFrame()

    df = pd.json_normalize(body[key])
    if df.empty:
        return df

    header = j.get("header", {})
    df["person_id"] = header.get("user_id", header.get("patient_id"))
    df["uuid"] = header.get("uuid")
    df["schema"] = header.get("schema_id", {}).get("name") if isinstance(header.get("schema_id"), dict) else None
    df["timezone"] = header.get("timezone")
    return df


def count_unique_days(ts: pd.Series | pd.DatetimeIndex, timezone: str | None = None) -> int:
    ts = pd.Series(pd.to_datetime(ts, errors="coerce", utc=True)).dropna()
    if len(ts) == 0:
        return 0
    if timezone:
        try:
            ts = ts.dt.tz_convert(timezone)
        except Exception:
            pass
    return int(ts.dt.floor("D").nunique())


def wear_minutes_from_heart_rate(
    hr_df: pd.DataFrame,
    time_col: str = "effective_time_frame.date_time",
    hr_col: str = "heart_rate.value",
    hr_min: float = 30,
    hr_max: float = 220,
) -> pd.DatetimeIndex:
    if hr_df is None or hr_df.empty:
        return pd.DatetimeIndex([])
    if time_col not in hr_df.columns or hr_col not in hr_df.columns:
        return pd.DatetimeIndex([])
    ts = pd.to_datetime(hr_df[time_col], errors="coerce", utc=True)
    hr = pd.to_numeric(hr_df[hr_col], errors="coerce")
    worn = ts.notna() & hr.between(hr_min, hr_max)
    return pd.DatetimeIndex(ts[worn].dt.floor("min").dropna().unique())


def pull_sleep_data(
    participant_id: int,
    path: Path,
    prefix: str,
    stage_col: str,
    start_col: str,
    end_col: str,
    wear_minutes: pd.DatetimeIndex | None = None,
    strict_missing: bool = False,
) -> pd.DataFrame:
    df = json_to_df(path, strict_missing=strict_missing)
    out = {"person_id": participant_id}
    if df.empty:
        if strict_missing:
            raise ValueError(f"Empty or missing sleep file for person_id={participant_id}: {path}")
        return pd.DataFrame([out])

    timezone = df["timezone"].dropna().iloc[0] if "timezone" in df.columns and df["timezone"].notna().any() else None

    ts_start = pd.to_datetime(df[start_col], errors="coerce", utc=True)
    ts_end = pd.to_datetime(df[end_col], errors="coerce", utc=True)

    hours = (ts_end - ts_start).dt.total_seconds() / 3600.0
    hours = pd.to_numeric(hours, errors="coerce").mask(lambda x: (x <= 0) | (x > SLEEP_INTERVAL_HOURS_MAX))

    if wear_minutes is not None and len(wear_minutes) > 0:
        mid = ts_start + (ts_end - ts_start) / 2
        keep = mid.dt.floor("min").isin(wear_minutes)
        ts_start = ts_start[keep]
        ts_end = ts_end[keep]
        hours = hours[keep]
        stage = df.loc[keep, stage_col].astype(str)
    else:
        stage = df[stage_col].astype(str)

    tmp = pd.DataFrame({"ts_start": ts_start, "ts_end": ts_end, "stage": stage, "hours": hours})
    tmp = tmp.dropna(subset=["ts_start", "ts_end", "stage", "hours"])

    if tmp.empty:
        out[f"{prefix}_ndays"] = 0
        out[f"{prefix}_valid_hours"] = 0
        return pd.DataFrame([out])

    if timezone:
        try:
            tmp["ts_start_local"] = tmp["ts_start"].dt.tz_convert(timezone)
            tmp["ts_end_local"] = tmp["ts_end"].dt.tz_convert(timezone)
        except Exception:
            tmp["ts_start_local"] = tmp["ts_start"]
            tmp["ts_end_local"] = tmp["ts_end"]
    else:
        tmp["ts_start_local"] = tmp["ts_start"]
        tmp["ts_end_local"] = tmp["ts_end"]

    covered_hours = []
    for s, e in zip(tmp["ts_start_local"], tmp["ts_end_local"]):
        start_hour = s.floor("h")
        end_hour = e.ceil("h") - pd.Timedelta(hours=1)
        if end_hour >= start_hour:
            covered_hours.extend(pd.date_range(start_hour, end_hour, freq="h"))

    out[f"{prefix}_valid_hours"] = len(pd.DatetimeIndex(covered_hours).unique())
    tmp["day"] = tmp["ts_start_local"].dt.floor("D")
    daily_stage = tmp.groupby(["day", "stage"])["hours"].sum().unstack("stage", fill_value=0.0)
    total = daily_stage.sum(axis=1)

    out[f"{prefix}_ndays"] = count_unique_days(tmp["ts_start"], timezone)

    total = pd.to_numeric(total, errors="coerce")
    total = total[np.isfinite(total)].dropna()
    out[f"{prefix}_total_median_hr"] = float(total.median()) if len(total) else None
    out[f"{prefix}_total_iqr_hr"] = float(total.quantile(0.75) - total.quantile(0.25)) if len(total) else None

    for s in daily_stage.columns:
        stage_vals = pd.to_numeric(daily_stage[s], errors="coerce")
        stage_vals = stage_vals[np.isfinite(stage_vals)].dropna()
        out[f"{prefix}_{s}_median_hr"] = float(stage_vals.median()) if len(stage_vals) else None
        out[f"{prefix}_{s}_iqr_hr"] = float(stage_vals.quantile(0.75) - stage_vals.quantile(0.25)) if len(stage_vals) else None

    return pd.DataFrame([out])


def pull_monitor_data(
    person_id: int,
    path: Path,
    value_col: str,
    prefix: str,
    time_col: str | None = None,
    start_col: str | None = None,
    end_col: str | None = None,
    wear_minutes: pd.DatetimeIndex | None = None,
    df: pd.DataFrame | None = None,
    min_samples_per_hour: int = 1,
    daily_agg: str | None = None,
    min_value: float | None = None,
    max_value: float | None = None,
    strict_missing: bool = False,
) -> pd.DataFrame:
    participant_df = df if df is not None else json_to_df(path, strict_missing=strict_missing)

    feature_cols = [
        "person_id",
        f"{prefix}_median",
        f"{prefix}_iqr",
        f"{prefix}_ndays",
        f"{prefix}_valid_hours",
    ]
    feature_row = pd.DataFrame([{c: None for c in feature_cols}])
    feature_row.loc[0, "person_id"] = person_id

    if participant_df.empty:
        if strict_missing:
            raise ValueError(f"Empty or missing data for {prefix}: person_id={person_id} path={path}")
        return feature_row

    timezone = participant_df["timezone"].dropna().iloc[0] if "timezone" in participant_df.columns and participant_df["timezone"].notna().any() else None
    ts = None

    if time_col and time_col in participant_df.columns:
        ts = pd.to_datetime(participant_df[time_col], errors="coerce", utc=True)
    elif start_col and end_col and start_col in participant_df.columns and end_col in participant_df.columns:
        ts_start = pd.to_datetime(participant_df[start_col], errors="coerce", utc=True)
        ts_end = pd.to_datetime(participant_df[end_col], errors="coerce", utc=True)
        ts = ts_start + (ts_end - ts_start) / 2
    elif start_col and start_col in participant_df.columns:
        ts = pd.to_datetime(participant_df[start_col], errors="coerce", utc=True)
    elif end_col and end_col in participant_df.columns:
        ts = pd.to_datetime(participant_df[end_col], errors="coerce", utc=True)

    if ts is None:
        return feature_row

    if wear_minutes is not None and len(wear_minutes) > 0:
        keep = ts.dt.floor("min").isin(wear_minutes)
        participant_df = participant_df.loc[keep].copy()
        ts = ts.loc[keep]

    ts = ts.dropna()
    if len(ts) == 0:
        feature_row.loc[0, f"{prefix}_ndays"] = 0
        feature_row.loc[0, f"{prefix}_valid_hours"] = 0
        return feature_row

    local_ts = ts.copy()
    if timezone:
        try:
            local_ts = ts.dt.tz_convert(timezone)
        except Exception:
            pass

    feature_row.loc[0, f"{prefix}_ndays"] = count_unique_days(ts, timezone)

    hour_counts = local_ts.dt.floor("h").value_counts()
    feature_row.loc[0, f"{prefix}_valid_hours"] = int((hour_counts >= min_samples_per_hour).sum())

    v = pd.to_numeric(participant_df[value_col], errors="coerce")
    if min_value is not None:
        v = v.mask(v < min_value)
    if max_value is not None:
        v = v.mask(v > max_value)

    tmp = pd.DataFrame({"ts": ts, "v": v}).dropna()
    if tmp.empty:
        return feature_row

    if daily_agg is not None:
        local_tmp_ts = tmp["ts"]
        if timezone:
            try:
                local_tmp_ts = tmp["ts"].dt.tz_convert(timezone)
            except Exception:
                pass
        day = local_tmp_ts.dt.floor("D")
        if daily_agg == "sum":
            vals = tmp.groupby(day)["v"].sum()
        elif daily_agg == "median":
            vals = tmp.groupby(day)["v"].median()
        else:
            raise ValueError("daily_agg must be None, 'sum', or 'median'")
        vals = vals.dropna()
    else:
        vals = tmp["v"]

    vals = pd.to_numeric(vals, errors="coerce")
    vals = vals[np.isfinite(vals)].dropna()

    feature_row.loc[0, f"{prefix}_median"] = float(vals.median()) if len(vals) else None
    feature_row.loc[0, f"{prefix}_iqr"] = float(vals.quantile(0.75) - vals.quantile(0.25)) if len(vals) else None
    return feature_row


def pull_environment_data(person_id: int, path: Path, prefix: str = "env", valid_ranges: dict | None = None, strict_missing: bool = False) -> pd.DataFrame:
    full_path = Path(path).expanduser()
    out = {"person_id": person_id}
    try:
        df = pd.read_csv(full_path, comment="#")
    except FileNotFoundError:
        if strict_missing:
            raise
        return pd.DataFrame([out])

    if df.empty or "ts" not in df.columns:
        if strict_missing:
            raise ValueError(f"Empty env data or missing ts column for person_id={person_id}: {path}")
        return pd.DataFrame([out])

    ts = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    df = df.assign(ts=ts).dropna(subset=["ts"])
    df["hour"] = df["ts"].dt.floor("h")
    out[f"{prefix}_valid_hours"] = int(df["hour"].nunique())

    use_cols = [c for c in ENV_VALUE_COLS if c in df.columns]
    if not use_cols:
        return pd.DataFrame([out])

    hourly = df.groupby("hour")[use_cols].median(numeric_only=True)
    valid_ranges = valid_ranges or {}

    for c in use_cols:
        v = pd.to_numeric(hourly[c], errors="coerce")
        min_value, max_value = valid_ranges.get(c, (None, None))
        if min_value is not None:
            v = v.mask(v < min_value)
        if max_value is not None:
            v = v.mask(v > max_value)
        v = v.replace([np.inf, -np.inf], np.nan).dropna()
        out[f"{prefix}_{c}_median"] = float(v.median()) if len(v) else None
        out[f"{prefix}_{c}_iqr"] = float(v.quantile(0.75) - v.quantile(0.25)) if len(v) else None

    return pd.DataFrame([out])
