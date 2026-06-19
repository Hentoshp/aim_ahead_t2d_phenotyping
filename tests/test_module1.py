from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import pytest

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from module1_processing.assemble import assemble
from module1_processing.cgm_features import summarize_cgm_series


def test_summarize_cgm_series_reports_broader_low_glucose_metrics():
    glucose = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=5, freq="5min", tz="UTC"),
            "value": [50.0, 60.0, 80.0, 120.0, 190.0],
        }
    )

    summary = summarize_cgm_series(glucose)

    expected_cv = np.std(glucose["value"], ddof=0) / np.mean(glucose["value"]) * 100.0
    assert np.isclose(summary["glycemic_cv"], expected_cv)
    assert np.isclose(summary["mean_glucose"], glucose["value"].mean())
    assert np.isclose(summary["time_in_range"], 2 / 5)
    assert np.isclose(summary["time_below_70"], 2 / 5)
    assert np.isclose(summary["time_below_54"], 1 / 5)


def test_assemble_supports_view_specific_cohorts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_root = tmp_path / "data"
    inter_dir = data_root / "processed" / "intermediates"
    processed_dir = data_root / "processed"
    qc_dir = data_root / "processed" / "qc_reports"
    inter_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    wearable = pd.DataFrame(
        {"heart_rate_median": [60.0, 62.0, 64.0]},
        index=["p1", "p2", "p3"],
    )
    environment = pd.DataFrame(
        {"env_temp_median": [20.0, 21.0, 22.0, 23.0]},
        index=["p1", "p2", "p3", "p4"],
    )
    cgm = pd.DataFrame(
        {
            "glycemic_cv": [18.0, 19.0, 20.0, 21.0],
            "mean_glucose": [110.0, 112.0, 114.0, 116.0],
            "time_in_range": [0.9, 0.91, 0.92, 0.93],
            "time_below_70": [0.01, 0.0, 0.02, 0.0],
            "time_below_54": [0.0, 0.0, 0.0, 0.0],
        },
        index=["p1", "p2", "p3", "p4"],
    )
    clinical = pd.DataFrame(
        {"hba1c": [5.5, 5.8, 6.1, 6.4], "diabetes_stage": [0, 1, 2, 3]},
        index=["p1", "p2", "p3", "p4"],
    )

    wearable.to_parquet(inter_dir / "wearable_features.parquet")
    environment.to_parquet(inter_dir / "environment_features.parquet")
    cgm.to_parquet(inter_dir / "cgm_features.parquet")
    clinical.to_parquet(inter_dir / "clinical_features.parquet")

    cfg = {
        "data": {
            "processed_path": "${AIREADI_DATA_PATH}/processed/",
            "intermediates_path": "${AIREADI_DATA_PATH}/processed/intermediates/",
            "qc_reports_path": "${AIREADI_DATA_PATH}/processed/qc_reports/",
        },
        "module1": {
            "normalization": "standard_scaler",
            "missing_strategy": "drop",
            "artifacts": {
                "level": "standard",
                "write_default_aliases": False,
                "save_view_raw_matrices": True,
                "save_common_raw_matrix": False,
            },
            "clustering_views": {
                "cohort_policy": "view_specific",
                "default_view": "wearable_environment",
                "views": {
                    "wearable": {"include_prefixes": ["heart_rate_"]},
                    "environment": {"include_prefixes": ["env_"]},
                    "wearable_environment": {"include_prefixes": ["heart_rate_", "env_"]},
                },
            },
        },
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("AIREADI_DATA_PATH", str(data_root))

    assemble(cfg_path)

    env_matrix = pd.read_parquet(processed_dir / "clustering_views" / "environment" / "clustering_matrix.parquet")
    env_outcomes = pd.read_parquet(processed_dir / "clustering_views" / "environment" / "outcome_matrix.parquet")
    combo_matrix = pd.read_parquet(processed_dir / "clustering_views" / "wearable_environment" / "clustering_matrix.parquet")
    combo_outcomes = pd.read_parquet(processed_dir / "clustering_views" / "wearable_environment" / "outcome_matrix.parquet")
    default_outcomes = pd.read_parquet(processed_dir / "outcome_matrix.parquet")

    assert len(env_matrix) == 4
    assert len(env_outcomes) == 4
    assert len(combo_matrix) == 3
    assert len(combo_outcomes) == 3
    assert len(default_outcomes) == 3
