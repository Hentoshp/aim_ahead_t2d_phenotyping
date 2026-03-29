# AI-Readi data processing
# Pierce Hentosh
import pandas as pd
import numpy as np
import os
import json

ENV_VALUE_COLS = [
    "pm1", "pm2.5", "pm4", "pm10",
    "hum", "temp", "voc", "nox",
    "lch0","lch1","lch2","lch3","lch6","lch7","lch8","lch9","lch10","lch11",
]

MONITOR_VALID_RANGES = {
    "cgm": (40, 400),
    "heart_rate": (25, 250),
    "oxygen_sat": (50, 100),
    "physical_activity": (0, None),
    "calories": (0, None),
    "respiratory_rate": (4, 60),
    "stress": (0, 100),
}

ENV_VALID_RANGES = {
    "pm1": (0, 65536),
    "pm2.5": (0, 65536),
    "pm4": (0, 65536),
    "pm10": (0, 65536),
    "hum": (0, 100),
    "temp": (-10, 50),
    "voc": (0, 500),
    "nox": (0, 500),
    "lch0": (0.0, 1.0),
    "lch1": (0.0, 1.0),
    "lch2": (0.0, 1.0),
    "lch3": (0.0, 1.0),
    "lch6": (0.0, 1.0),
    "lch7": (0.0, 1.0),
    "lch8": (0.0, 1.0),
    "lch9": (0.0, 1.0),
    "lch10": (0.0, 1.0),
    "lch11": (0.0, 1.0),
}

SLEEP_INTERVAL_HOURS_MAX = 24


def pull_participants() -> pd.DataFrame:
    """Pull relevant participant data from AIM-AHEAD"""

    participants = pd.read_csv(ROOT + 'participants.tsv', sep='\t')
    cohort_columns = ['clinical_data', 'environment', 'wearable_activity_monitor', 'wearable_blood_glucose']
    filtered_participants = participants[participants[cohort_columns].all(axis=1)]

    dropped_columns = cohort_columns + [
        'clinical_site', 'study_visit_date', 'cardiac_ecg', 'retinal_flio',
        'retinal_oct', 'retinal_octa', 'retinal_photography'
    ]
    filtered_participants = filtered_participants.drop(columns=dropped_columns)
    filtered_participants = filtered_participants.reset_index(drop=True)

    return filtered_participants


def json_to_df(path: str, key=None) -> pd.DataFrame:
    """
    Process JSON file into a dataframe
    """
    try:
        path = os.path.expanduser(path)
        with open(path) as f:
            j = json.load(f)
    except FileNotFoundError:
        return pd.DataFrame()

    body = j["body"]
    if key is None:
        key = next(k for k, v in body.items() if isinstance(v, list))

    df = pd.json_normalize(body[key])
    if df.empty:
        return df
    else:
        header = j.get("header", {})
        df["person_id"] = header.get("user_id", header.get("patient_id"))
        df["uuid"] = header.get("uuid")
        df["schema"] = header.get("schema_id", {}).get("name")
        df["timezone"] = header.get("timezone")

    return df


def count_unique_days(ts: pd.Series | pd.DatetimeIndex, timezone: str | None = None) -> int:
    """
    Standardized n_days computation used across all wearable modalities.
    Counts unique calendar days from valid timestamps.
    """
    ts = pd.Series(pd.to_datetime(ts, errors="coerce", utc=True)).dropna()
    if len(ts) == 0:
        return 0

    if timezone:
        try:
            ts = ts.dt.tz_convert(timezone)
        except Exception:
            pass

    return int(ts.dt.floor("D").nunique())


def wear_minutes_from_heart_rate(hr_df: pd.DataFrame,
                                 time_col: str = "effective_time_frame.date_time",
                                 hr_col: str = "heart_rate.value",
                                 hr_min: float = 30,
                                 hr_max: float = 220) -> pd.DatetimeIndex:
    """Calculate wearable minutes from heart rate, i.e. time where heart rate is physiologically possible"""
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
    path: str,
    prefix: str,
    stage_col: str,
    start_col: str,
    end_col: str,
    wear_minutes: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """
    Minimal sleep wrapper:
    - loads sleep df
    - converts each interval to hours (end-start), drops <=0
    - (optional) filters intervals to HR-worn minutes using midpoint
    - aggregates to daily totals by stage
    - returns median/IQR across days for each stage + total
    """
    df = json_to_df(ROOT + path)

    out = {"person_id": participant_id}
    if df.empty:
        return pd.DataFrame([out])

    timezone = None
    if "timezone" in df.columns and df["timezone"].notna().any():
        timezone = df["timezone"].dropna().iloc[0]

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

    tmp = pd.DataFrame({
        "ts_start": ts_start,
        "ts_end": ts_end,
        "stage": stage,
        "hours": hours
    }).dropna(subset=["ts_start", "ts_end", "stage", "hours"])

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
    bad_total = ~np.isfinite(total)
    if bad_total.any():
        print(f"[WARN] Non-finite values in {prefix}_total | person_id={participant_id} | path={path} | n_bad={int(bad_total.sum())}")
    total = total[~bad_total].dropna()

    out[f"{prefix}_total_median_hr"] = float(total.median()) if len(total) else None
    out[f"{prefix}_total_iqr_hr"] = float(total.quantile(0.75) - total.quantile(0.25)) if len(total) else None

    for s in daily_stage.columns:
        stage_vals = pd.to_numeric(daily_stage[s], errors="coerce")
        bad_stage = ~np.isfinite(stage_vals)
        if bad_stage.any():
            print(f"[WARN] Non-finite values in {prefix}_{s} | person_id={participant_id} | path={path} | n_bad={int(bad_stage.sum())}")
        stage_vals = stage_vals[~bad_stage].dropna()

        out[f"{prefix}_{s}_median_hr"] = float(stage_vals.median()) if len(stage_vals) else None
        out[f"{prefix}_{s}_iqr_hr"] = float(stage_vals.quantile(0.75) - stage_vals.quantile(0.25)) if len(stage_vals) else None

    return pd.DataFrame([out])


def pull_monitor_data(
    person_id: int,
    path: str,
    value_col: str,
    prefix: str,
    time_col: str | None = None,
    start_col: str | None = None,
    end_col: str | None = None,
    wear_minutes: pd.DatetimeIndex | None = None,
    df: pd.DataFrame | None = None,
    min_samples_per_hour: int = 1,
    daily_agg: str | None = None,   # None | "sum" | "median"
    min_value: float | None = None,
    max_value: float | None = None,
) -> pd.DataFrame:
    """
    Features: n_days, median, IQR, valid_hours (hours with >= min_samples_per_hour samples)
    - Interval streams use midpoint timestamps (better for wear filtering + binning).
    - daily_agg can summarize per-day first (e.g., steps/day) before median/IQR.
    """
    participant_df = df if df is not None else json_to_df(ROOT + path)

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
        return feature_row

    timezone = None
    if "timezone" in participant_df.columns and participant_df["timezone"].notna().any():
        timezone = participant_df["timezone"].dropna().iloc[0]

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
    bad = ~np.isfinite(vals)
    if bad.any():
        print(f"[WARN] Non-finite values in {prefix} | person_id={person_id} | path={path} | value_col={value_col} | n_bad={int(bad.sum())}")
    vals = vals[~bad].dropna()

    feature_row.loc[0, f"{prefix}_median"] = float(vals.median()) if len(vals) else None
    feature_row.loc[0, f"{prefix}_iqr"] = float(vals.quantile(0.75) - vals.quantile(0.25)) if len(vals) else None

    return feature_row


def pull_environment_data(person_id: int, path: str, prefix: str = "env") -> pd.DataFrame:
    """
    Minimal env features per participant:
      - {prefix}_valid_hours  (hours with any data)
      - {prefix}_{col}_median, {prefix}_{col}_iqr  (computed from hourly medians)
    """
    full_path = os.path.expanduser(ROOT + path)
    out = {"person_id": person_id}

    try:
        df = pd.read_csv(full_path, comment="#")
    except FileNotFoundError:
        return pd.DataFrame([out])

    if df.empty or "ts" not in df.columns:
        return pd.DataFrame([out])

    ts = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    df = df.assign(ts=ts).dropna(subset=["ts"])

    # Hourly bins
    df["hour"] = df["ts"].dt.floor("h")

    # Valid hours = hours with at least 1 row (or use subset of cols if you want stricter)
    out[f"{prefix}_valid_hours"] = int(df["hour"].nunique())

    # Hourly medians per variable (reduces weighting artifacts)
    use_cols = [c for c in ENV_VALUE_COLS if c in df.columns]
    if not use_cols:
        return pd.DataFrame([out])

    hourly = df.groupby("hour")[use_cols].median(numeric_only=True)

    # Summaries across hours
    for c in use_cols:
        v = pd.to_numeric(hourly[c], errors="coerce")

        min_value, max_value = ENV_VALID_RANGES.get(c, (None, None))
        if min_value is not None:
            v = v.mask(v < min_value)
        if max_value is not None:
            v = v.mask(v > max_value)

        n_nan = int(v.isna().sum())
        n_inf = int(np.isinf(v).sum())

        if n_nan or n_inf:
            print(
                f"[WARN] Invalid values in {prefix}_{c} | person_id={person_id} | path={path} "
                f"| n_nan={n_nan} | n_inf={n_inf}"
            )

        v = v.replace([np.inf, -np.inf], np.nan).dropna()
        out[f"{prefix}_{c}_median"] = float(v.median()) if len(v) else None
        out[f"{prefix}_{c}_iqr"] = float(v.quantile(0.75) - v.quantile(0.25)) if len(v) else None

    return pd.DataFrame([out])


# pull participant data
participants_df = pull_participants()
id_list = participants_df['person_id']

glucose_rows = []
heartrate_rows = []
oxygen_sat_rows = []
activity_rows = []
calories_rows = []
respiratory_rate_rows = []
sleep_rows = []
stress_rows = []
env_rows = []

# loop over monitor data

for id in id_list:

    # CGM (independent of Garmin wear mask)
    glucose_path = f"wearable_blood_glucose/continuous_glucose_monitoring/dexcom_g6/{str(id)}/{str(id)}_DEX.json"
    glucose_rows.append(pull_monitor_data(
        id, glucose_path, "blood_glucose.value", "cgm",
        start_col="effective_time_frame.time_interval.start_date_time",
        end_col="effective_time_frame.time_interval.end_date_time",
        min_value=MONITOR_VALID_RANGES["cgm"][0],
        max_value=MONITOR_VALID_RANGES["cgm"][1],
    ))

    # Heart rate: load once to compute wear minutes
    heartrate_path = f"wearable_activity_monitor/heart_rate/garmin_vivosmart5/{str(id)}/{str(id)}_heartrate.json"
    hr_df = json_to_df(ROOT + heartrate_path)
    wear_mins = wear_minutes_from_heart_rate(hr_df)

    # HR summary restricted to worn minutes (so valid_hours reflects wear)
    heartrate_rows.append(pull_monitor_data(
        id, heartrate_path, "heart_rate.value", "heart_rate",
        time_col="effective_time_frame.date_time",
        wear_minutes=wear_mins,
        min_samples_per_hour=10,
        min_value=MONITOR_VALID_RANGES["heart_rate"][0],
        max_value=MONITOR_VALID_RANGES["heart_rate"][1],
    ))

    # Oxygen sat filtered to worn minutes
    oxygen_sat_path = f"wearable_activity_monitor/oxygen_saturation/garmin_vivosmart5/{str(id)}/{str(id)}_oxygensaturation.json"
    oxygen_sat_rows.append(pull_monitor_data(
        id, oxygen_sat_path, "oxygen_saturation.value", "oxygen_sat",
        time_col="effective_time_frame.date_time",
        wear_minutes=wear_mins,
        min_value=MONITOR_VALID_RANGES["oxygen_sat"][0],
        max_value=MONITOR_VALID_RANGES["oxygen_sat"][1],
    ))

    # activity
    activity_path = f"wearable_activity_monitor/physical_activity/garmin_vivosmart5/{str(id)}/{str(id)}_activity.json"
    activity_rows.append(pull_monitor_data(
        id, activity_path, "base_movement_quantity.value", "physical_activity",
        start_col="effective_time_frame.time_interval.start_date_time",
        end_col="effective_time_frame.time_interval.end_date_time",
        wear_minutes=wear_mins,
        daily_agg="sum",
        min_value=MONITOR_VALID_RANGES["physical_activity"][0],
        max_value=MONITOR_VALID_RANGES["physical_activity"][1],
    ))

    # activity calories burned
    calories_path = f"wearable_activity_monitor/physical_activity_calorie/garmin_vivosmart5/{str(id)}/{str(id)}_calorie.json"
    calories_rows.append(pull_monitor_data(
        id, calories_path, "calories_value.value", "calories",
        time_col="effective_time_frame.date_time",
        wear_minutes=wear_mins,
        daily_agg="sum",
        min_value=MONITOR_VALID_RANGES["calories"][0],
        max_value=MONITOR_VALID_RANGES["calories"][1],
    ))

    # respiratory
    respiratory_rate_path = f"wearable_activity_monitor/respiratory_rate/garmin_vivosmart5/{str(id)}/{str(id)}_respiratoryrate.json"
    respiratory_rate_rows.append(pull_monitor_data(
        id, respiratory_rate_path, "respiratory_rate.value", "respiratory_rate",
        time_col="effective_time_frame.date_time",
        wear_minutes=wear_mins,
        min_value=MONITOR_VALID_RANGES["respiratory_rate"][0],
        max_value=MONITOR_VALID_RANGES["respiratory_rate"][1],
    ))

    # sleep
    sleep_path = f"wearable_activity_monitor/sleep/garmin_vivosmart5/{str(id)}/{str(id)}_sleep.json"
    sleep_rows.append(pull_sleep_data(
        id, sleep_path, "sleep", "sleep_stage_state",
        "effective_time_frame.time_interval.start_date_time",
        "effective_time_frame.time_interval.end_date_time",
        wear_minutes=wear_mins
    ))

    # stress
    stress_path = f"wearable_activity_monitor/stress/garmin_vivosmart5/{str(id)}/{str(id)}_stress.json"
    stress_rows.append(pull_monitor_data(
        id, stress_path, "stress.value",
        "stress", time_col="effective_time_frame.date_time",
        wear_minutes=wear_mins,
        min_value=MONITOR_VALID_RANGES["stress"][0],
        max_value=MONITOR_VALID_RANGES["stress"][1],
    ))

    # env
    env_path = f"environment/environmental_sensor/leelab_anura/{str(id)}/{str(id)}_ENV.csv"
    env_rows.append(pull_environment_data(id, env_path, prefix="env"))


glucose_df = pd.concat(glucose_rows, ignore_index=True)
heartrate_df = pd.concat(heartrate_rows, ignore_index=True)
oxygen_sat_df = pd.concat(oxygen_sat_rows, ignore_index=True)
activity_df = pd.concat(activity_rows, ignore_index=True)
calorie_df = pd.concat(calories_rows, ignore_index=True)
respiratory_rate_df = pd.concat(respiratory_rate_rows, ignore_index=True)
sleep_df = pd.concat(sleep_rows, ignore_index=True)
stress_df = pd.concat(stress_rows, ignore_index=True)
env_df = pd.concat(env_rows, ignore_index=True)

dfs = [glucose_df, heartrate_df, oxygen_sat_df, activity_df, calorie_df,
       respiratory_rate_df, sleep_df, stress_df, env_df]

# start from participant roster (keeps everyone who passed your cohort filter)
final_df = participants_df[["person_id"]].copy()

# merge all feature tables
for d in dfs:
    final_df = final_df.merge(d, on="person_id", how="left")

final_df.to_csv("aim_ahead_data.csv", index=False)