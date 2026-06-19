import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from module3_bayesian.pipeline import (
    CONTINUOUS_DUAL_MODEL,
    DIABETES_STAGE_RAW_COL,
    PROBABILITY_TOLERANCE,
    THRESHOLD_MODEL,
    normalize_diabetes_stage_series,
    read_analysis_inputs,
    run_pipeline,
    validate_probability_matrix,
)


def _prepare_pymc_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MPLCONFIGDIR", "/tmp")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / ".cache"))
    monkeypatch.setenv("PYTENSOR_FLAGS", f"base_compiledir={tmp_path / '.pytensor'}")
    pytest.importorskip("pymc", reason="pymc is required for Module 3 fit tests")


def _write_config(
    base_dir: Path,
    primary_model: str = CONTINUOUS_DUAL_MODEL,
    run_comparison_models: bool = False,
    draws: int = 100,
    tune: int = 100,
    chains: int = 2,
) -> Path:
    secondary_outcomes = ["time_below_70"] if primary_model == CONTINUOUS_DUAL_MODEL else []
    cfg = {
        "data": {
            "processed_path": "${AIREADI_DATA_PATH}/processed/",
            "artifacts_path": "${AIREADI_DATA_PATH}/artifacts/",
        },
        "module2": {
            "random_seed": 42,
        },
        "module3": {
            "primary_model": primary_model,
            "severity_covariates": ["hba1c", "diabetes_stage"],
            "primary_outcome": "glycemic_cv",
            "secondary_outcomes": secondary_outcomes,
            "cv_clinical_threshold": 36.0,
            "proportion_epsilon": 1e-4,
            "run_comparison_models": run_comparison_models,
            "sampling": {
                "draws": draws,
                "tune": tune,
                "chains": chains,
                "target_accept": 0.9,
            },
        },
    }
    cfg_path = base_dir / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return cfg_path


def _write_direct_inputs(
    base_dir: Path,
    n: int = 60,
    write_view_specific_outcome: bool = False,
) -> tuple[Path, Path, pd.DataFrame, pd.DataFrame]:
    processed_dir = base_dir / "processed"
    artifacts_dir = base_dir / "artifacts" / "module2" / "wearable_environment" / "stability_v1"
    view_processed_dir = processed_dir / "clustering_views" / "wearable_environment"
    processed_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    view_processed_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(7)
    participant_ids = [f"p{i:03d}" for i in range(n)]
    stages = rng.choice([0, 1, 2, 3], size=n, p=[0.15, 0.30, 0.30, 0.25])
    hba1c = rng.normal(loc=6.9 + 0.45 * stages, scale=0.45, size=n)

    raw_membership = rng.dirichlet(alpha=np.array([2.5, 2.0, 1.8]), size=n)
    membership_df = pd.DataFrame(raw_membership, index=participant_ids, columns=["pi_1", "pi_2", "pi_3"])

    stable_logit = (
        1.1
        - 0.55 * (hba1c - hba1c.mean())
        - 0.35 * stages
        + 0.8 * membership_df["pi_1"].to_numpy()
        - 0.5 * membership_df["pi_2"].to_numpy()
    )
    p_stable = 1.0 / (1.0 + np.exp(-stable_logit))
    stable = rng.binomial(1, p_stable, size=n)
    glycemic_cv = np.where(stable == 1, rng.normal(29.0, 3.0, size=n), rng.normal(43.0, 3.0, size=n))

    tbr70_latent = (
        -2.6
        + 0.30 * (hba1c - hba1c.mean())
        + 0.28 * stages
        - 0.80 * membership_df["pi_1"].to_numpy()
        + 0.55 * membership_df["pi_2"].to_numpy()
        + rng.normal(0.0, 0.35, size=n)
    )
    time_below_70 = np.clip((1.0 / (1.0 + np.exp(-tbr70_latent))) * 0.22, 0.0, 0.35)
    time_below_70[time_below_70 < 0.015] = 0.0
    time_below_54 = np.clip(time_below_70 * rng.uniform(0.10, 0.45, size=n), 0.0, 1.0)
    time_below_54[time_below_54 < 0.003] = 0.0

    time_in_range = np.clip(
        0.88 - 0.004 * (glycemic_cv - 30.0) - 0.60 * time_below_70 + rng.normal(0.0, 0.03, size=n),
        0.20,
        0.98,
    )

    outcome_df = pd.DataFrame(
        {
            "glycemic_cv": glycemic_cv,
            "mean_glucose": rng.normal(140, 18, size=n),
            "time_in_range": time_in_range,
            "time_below_70": time_below_70,
            "time_below_54": time_below_54,
            "hba1c": hba1c,
            "diabetes_stage": stages,
        },
        index=participant_ids,
    )

    outcome_path = processed_dir / "outcome_matrix.parquet"
    membership_path = artifacts_dir / "membership_matrix.parquet"
    outcome_df.to_parquet(outcome_path)
    if write_view_specific_outcome:
        outcome_df.to_parquet(view_processed_dir / "outcome_matrix.parquet")
    membership_df.to_parquet(membership_path)
    return outcome_path, membership_path, outcome_df, membership_df


def test_validate_probability_matrix_rejects_rows_not_summing_to_one():
    bad = pd.DataFrame({"pi_1": [0.7, 0.4], "pi_2": [0.4, 0.4], "pi_3": [0.1, 0.1]})
    with pytest.raises(ValueError, match="sum to 1"):
        validate_probability_matrix(bad)


def test_read_analysis_inputs_builds_expected_columns(tmp_path: Path):
    outcome_path, membership_path, outcome_df, _ = _write_direct_inputs(tmp_path, n=16)
    artifacts = read_analysis_inputs(
        outcome_matrix_path=outcome_path,
        membership_matrix_path=membership_path,
        threshold=36.0,
        required_outcome_cols={"glycemic_cv", "time_below_70"},
    )

    assert "stable" in artifacts.analysis_df.columns
    assert "unstable" in artifacts.analysis_df.columns
    assert "hba1c_z" in artifacts.analysis_df.columns
    assert DIABETES_STAGE_RAW_COL in artifacts.analysis_df.columns
    assert "top_cluster" in artifacts.analysis_df.columns
    assert "max_membership" in artifacts.analysis_df.columns
    assert "time_below_70" in artifacts.analysis_df.columns
    assert "time_below_54" in artifacts.analysis_df.columns
    assert artifacts.reference_cluster in artifacts.probability_columns
    assert artifacts.reference_cluster not in artifacts.predictor_probability_columns

    expected_stable = (outcome_df["glycemic_cv"] < 36.0).astype(int)
    assert artifacts.analysis_df["stable"].equals(expected_stable.loc[artifacts.analysis_df.index])
    assert np.allclose(
        artifacts.analysis_df[artifacts.probability_columns].sum(axis=1).to_numpy(),
        np.ones(len(artifacts.analysis_df)),
        atol=PROBABILITY_TOLERANCE,
    )

    diag = artifacts.diagnostics_df
    stable_count = int(expected_stable.sum())
    unstable_count = int((1 - expected_stable).sum())
    stable_row = diag[(diag["section"] == "stable_outcome") & (diag["group"] == "stable")]
    unstable_row = diag[(diag["section"] == "stable_outcome") & (diag["group"] == "unstable")]
    tbr_zero_row = diag[
        (diag["section"] == "outcome_distribution")
        & (diag["group"] == "time_below_70")
        & (diag["metric"] == "zero_rate")
    ]
    assert int(stable_row.iloc[0]["value"]) == stable_count
    assert int(unstable_row.iloc[0]["value"]) == unstable_count
    assert float(tbr_zero_row.iloc[0]["value_numeric"]) >= 0.0


def test_normalize_diabetes_stage_series_accepts_string_labels():
    raw = pd.Series(
        [
            "control",
            "prediabetes",
            "non_insulin_dependent",
            "insulin_dependent",
        ],
        index=["a", "b", "c", "d"],
    )
    normalized = normalize_diabetes_stage_series(raw)
    assert normalized.tolist() == [0, 1, 2, 3]


def test_read_analysis_inputs_accepts_string_stage_labels(tmp_path: Path):
    outcome_path, membership_path, outcome_df, _ = _write_direct_inputs(tmp_path, n=12)
    stage_labels = np.array(
        ["control", "prediabetes", "non_insulin_dependent", "insulin_dependent"] * 3,
        dtype=object,
    )
    outcome_df["diabetes_stage"] = stage_labels[: len(outcome_df)]
    outcome_df.to_parquet(outcome_path)

    artifacts = read_analysis_inputs(
        outcome_matrix_path=outcome_path,
        membership_matrix_path=membership_path,
        threshold=36.0,
        required_outcome_cols={"glycemic_cv", "time_below_70"},
    )

    assert set(artifacts.analysis_df["diabetes_stage"].unique().tolist()) == {0, 1, 2, 3}
    assert artifacts.analysis_df[DIABETES_STAGE_RAW_COL].iloc[0] == "control"


def test_run_pipeline_rejects_comparison_models_before_sampling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _write_direct_inputs(tmp_path, n=12)
    cfg_path = _write_config(tmp_path, run_comparison_models=True)
    monkeypatch.setenv("AIREADI_DATA_PATH", str(tmp_path))

    with pytest.raises(NotImplementedError, match="run_comparison_models"):
        run_pipeline(cfg_path=cfg_path, view="wearable_environment", experiment="stability_v1")


def test_run_pipeline_continuous_mode_writes_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _prepare_pymc_runtime(tmp_path, monkeypatch)
    _write_direct_inputs(tmp_path, n=60)
    cfg_path = _write_config(tmp_path, primary_model=CONTINUOUS_DUAL_MODEL, run_comparison_models=False)
    monkeypatch.setenv("AIREADI_DATA_PATH", str(tmp_path))

    run_summary_path = run_pipeline(cfg_path=cfg_path, view="wearable_environment", experiment="stability_v1")

    out_dir = run_summary_path.parent
    assert (out_dir / "analysis_dataset.parquet").exists()
    assert (out_dir / "diagnostics_table.parquet").exists()
    assert (out_dir / "participant_predictions.parquet").exists()
    assert (out_dir / "model_summary.parquet").exists()
    assert (out_dir / "cluster_stage_predictions.parquet").exists()
    assert (out_dir / "sampling_diagnostics.json").exists()
    assert (out_dir / "posterior_predictive_summary.json").exists()
    assert (out_dir / "inference_data.nc").exists()

    participant_predictions = pd.read_parquet(out_dir / "participant_predictions.parquet")
    assert {
        "glycemic_cv_pred_mean",
        "time_below_70_pred_mean",
        "top_cluster",
        "max_membership",
    }.issubset(participant_predictions.columns)
    assert ((participant_predictions["time_below_70_pred_mean"] >= 0) & (participant_predictions["time_below_70_pred_mean"] <= 1)).all()

    diagnostics = json.loads((out_dir / "sampling_diagnostics.json").read_text())
    assert "divergences" in diagnostics
    assert "max_rhat" in diagnostics

    run_summary = json.loads(run_summary_path.read_text())
    assert run_summary["primary_model"] == CONTINUOUS_DUAL_MODEL
    assert "target_estimands" in run_summary


def test_run_pipeline_threshold_mode_writes_probability_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _prepare_pymc_runtime(tmp_path, monkeypatch)
    _write_direct_inputs(tmp_path, n=60)
    cfg_path = _write_config(
        tmp_path,
        primary_model=THRESHOLD_MODEL,
        run_comparison_models=False,
        draws=250,
        tune=250,
    )
    monkeypatch.setenv("AIREADI_DATA_PATH", str(tmp_path))

    run_summary_path = run_pipeline(cfg_path=cfg_path, view="wearable_environment", experiment="stability_v1")

    out_dir = run_summary_path.parent
    participant_predictions = pd.read_parquet(out_dir / "participant_predictions.parquet")
    assert {"p_stable_mean", "p_unstable_mean", "top_cluster", "max_membership"}.issubset(participant_predictions.columns)
    assert ((participant_predictions["p_stable_mean"] >= 0) & (participant_predictions["p_stable_mean"] <= 1)).all()

    run_summary = json.loads(run_summary_path.read_text())
    assert run_summary["primary_model"] == THRESHOLD_MODEL
    assert "target_probability" in run_summary


def test_run_pipeline_uses_view_specific_outcome_matrix_when_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _prepare_pymc_runtime(tmp_path, monkeypatch)
    outcome_path, membership_path, outcome_df, _ = _write_direct_inputs(
        tmp_path,
        n=60,
        write_view_specific_outcome=True,
    )
    view_outcome_path = tmp_path / "processed" / "clustering_views" / "wearable_environment" / "outcome_matrix.parquet"
    outcome_path.unlink()

    cfg_path = _write_config(tmp_path, primary_model=THRESHOLD_MODEL, run_comparison_models=False, draws=250, tune=250)
    monkeypatch.setenv("AIREADI_DATA_PATH", str(tmp_path))

    run_summary_path = run_pipeline(cfg_path=cfg_path, view="wearable_environment", experiment="stability_v1")
    run_summary = json.loads(run_summary_path.read_text())

    assert run_summary["source_outcome_matrix_path"] == str(view_outcome_path)
