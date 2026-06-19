"""Module 3 Bayesian modeling pipeline.

Supports two modeling modes:
- ``threshold``: logistic regression for ``P(glycemic_cv < threshold | X)``
- ``continuous_dual``: continuous regressions for expected ``glycemic_cv`` and
  expected ``time_below_70`` given the same covariates
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from parquet_utils import read_parquet_with_compat_hint


THRESHOLD_MODEL = "threshold"
CONTINUOUS_DUAL_MODEL = "continuous_dual"
PRIMARY_OUTCOME_GLYCEMIC_CV = "glycemic_cv"
SECONDARY_OUTCOME_TIME_BELOW_70 = "time_below_70"
OPTIONAL_OUTCOME_TIME_BELOW_54 = "time_below_54"
STABLE_LABEL = "stable"
UNSTABLE_LABEL = "unstable"
PREDICTOR_HBA1C_Z = "hba1c_z"
DIABETES_STAGE_RAW_COL = "diabetes_stage_raw"
TOP_CLUSTER_COL = "top_cluster"
MAX_MEMBERSHIP_COL = "max_membership"
MEMBERSHIP_MARGIN_COL = "membership_margin"
BORDERLINE_MAX_THRESHOLD = 0.60
BORDERLINE_MARGIN_THRESHOLD = 0.20
PROBABILITY_TOLERANCE = 1e-6
DEFAULT_RANDOM_SEED = 42
R_HAT_THRESHOLD = 1.05
ESS_BULK_THRESHOLD = 50.0
DEFAULT_PROPORTION_EPSILON = 1e-4
STAGE_LABEL_ALIASES = {
    0: {
        "0",
        "control",
        "healthy",
        "no_diabetes",
        "nondiabetic",
        "non_diabetic",
        "non_diabetic_control",
        "non_t2d",
        "no_t2d",
        "normoglycemic",
        "normal",
    },
    1: {
        "1",
        "prediabetes",
        "pre_diabetes",
        "prediabetic",
        "lifestyle_controlled",
        "lifestyle",
    },
    2: {
        "2",
        "oral",
        "oral_medication",
        "oral_medications",
        "non_insulin",
        "non_insulin_dependent",
        "non_insulin_injectable",
        "oral_non_insulin_injectable",
        "oral_or_non_insulin_injectable",
        "non_insulin_dependent_t2d",
        "t2d_non_insulin",
    },
    3: {
        "3",
        "insulin",
        "insulin_dependent",
        "insulin_controlled",
        "insulin_dependent_t2d",
        "t2d_insulin",
    },
}


@dataclass
class ResolvedPaths:
    processed_path: Path
    artifacts_path: Path
    outcome_matrix_path: Path


@dataclass
class SourceResolution:
    source_view: str
    source_experiment: str
    source_artifacts_path: Path
    source_outcome_matrix_path: Path
    output_dir: Path
    mode: str
    slot: str | None = None


@dataclass
class AnalysisArtifacts:
    analysis_df: pd.DataFrame
    diagnostics_df: pd.DataFrame
    probability_columns: list[str]
    predictor_probability_columns: list[str]
    reference_cluster: str
    reference_stage: int
    observed_stages: list[int]
    hba1c_mean: float
    hba1c_std: float
    threshold: float


@dataclass
class ContinuousOutcomeData:
    glycemic_cv_z: np.ndarray
    glycemic_cv_mean: float
    glycemic_cv_std: float
    time_below_70_logit_z: np.ndarray
    time_below_70_logit_mean: float
    time_below_70_logit_std: float
    proportion_epsilon: float


@dataclass
class ModelFitArtifacts:
    idata: Any
    model_summary_df: pd.DataFrame
    participant_predictions: pd.DataFrame
    cluster_stage_predictions: pd.DataFrame
    sampling_diagnostics: dict[str, Any]
    posterior_predictive: dict[str, Any]
    run_metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Module 3 Bayesian modeling pipeline")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--slot", default=None, help="Selected slot name, e.g. primary or sensitivity")
    parser.add_argument("--view", default=None, help="Source view name when not using --slot")
    parser.add_argument("--experiment", default=None, help="Source experiment name when not using --slot")
    return parser.parse_args()


def load_config(cfg_path: Path) -> dict[str, Any]:
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def resolve_paths(cfg: dict[str, Any]) -> ResolvedPaths:
    load_dotenv()
    data_root = os.getenv("AIREADI_DATA_PATH")
    if not data_root:
        raise EnvironmentError("AIREADI_DATA_PATH not set; define it in .env")

    base = Path(data_root).expanduser()
    processed_path = _substitute_env(cfg["data"]["processed_path"], base)
    artifacts_path = _substitute_env(cfg["data"]["artifacts_path"], base)
    return ResolvedPaths(
        processed_path=processed_path,
        artifacts_path=artifacts_path,
        outcome_matrix_path=processed_path / "outcome_matrix.parquet",
    )


def _substitute_env(template: str, data_root: Path) -> Path:
    return Path(template.replace("${AIREADI_DATA_PATH}", str(data_root))).expanduser()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_outcome_matrix_path(processed_path: Path, view_name: str) -> Path:
    view_specific_path = processed_path / "clustering_views" / str(view_name) / "outcome_matrix.parquet"
    if view_specific_path.exists():
        return view_specific_path
    return processed_path / "outcome_matrix.parquet"


def validate_module3_config(cfg: dict[str, Any]) -> dict[str, Any]:
    module3_cfg = cfg.get("module3", {})
    primary_model = str(module3_cfg.get("primary_model", CONTINUOUS_DUAL_MODEL))
    if primary_model not in {THRESHOLD_MODEL, CONTINUOUS_DUAL_MODEL}:
        raise NotImplementedError(
            f"module3.primary_model={primary_model!r} is not implemented; "
            f"supported models are {THRESHOLD_MODEL!r} and {CONTINUOUS_DUAL_MODEL!r}."
        )

    if bool(module3_cfg.get("run_comparison_models", False)):
        raise NotImplementedError(
            "module3.run_comparison_models=true is not implemented; set it to false."
        )

    primary_outcome = str(module3_cfg.get("primary_outcome", PRIMARY_OUTCOME_GLYCEMIC_CV))
    if primary_outcome != PRIMARY_OUTCOME_GLYCEMIC_CV:
        raise NotImplementedError(
            f"module3.primary_outcome={primary_outcome!r} is not implemented; "
            f"only {PRIMARY_OUTCOME_GLYCEMIC_CV!r} is supported."
        )

    expected_covariates = {"hba1c", "diabetes_stage"}
    configured_covariates = set(module3_cfg.get("severity_covariates", []))
    if configured_covariates != expected_covariates:
        raise ValueError(
            "module3.severity_covariates must be exactly ['hba1c', 'diabetes_stage']."
        )

    secondary_outcomes = list(module3_cfg.get("secondary_outcomes", []))
    if primary_model == CONTINUOUS_DUAL_MODEL and secondary_outcomes != [SECONDARY_OUTCOME_TIME_BELOW_70]:
        raise ValueError(
            "module3.secondary_outcomes must be exactly ['time_below_70'] for the continuous_dual model."
        )
    if primary_model == THRESHOLD_MODEL and secondary_outcomes:
        raise NotImplementedError(
            "module3.secondary_outcomes is only supported for the continuous_dual model."
        )
    return module3_cfg


def required_outcome_columns(module3_cfg: dict[str, Any]) -> set[str]:
    required = {PRIMARY_OUTCOME_GLYCEMIC_CV}
    if str(module3_cfg.get("primary_model", CONTINUOUS_DUAL_MODEL)) == CONTINUOUS_DUAL_MODEL:
        required.add(SECONDARY_OUTCOME_TIME_BELOW_70)
    return required


def resolve_source(
    cfg: dict[str, Any],
    slot: str | None = None,
    view: str | None = None,
    experiment: str | None = None,
) -> SourceResolution:
    paths = resolve_paths(cfg)
    if slot:
        manifest_path = paths.artifacts_path / "module2" / "selected" / str(slot) / "selection_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Selection manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        source_view = str(manifest["source_view"])
        source_experiment = str(manifest["source_experiment"])
        source_artifacts_path = Path(str(manifest["source_artifacts_path"]))
        manifest_outcome_path = manifest.get("source_outcome_matrix_path")
        if manifest_outcome_path:
            source_outcome_matrix_path = Path(str(manifest_outcome_path))
        else:
            source_outcome_matrix_path = resolve_outcome_matrix_path(paths.processed_path, source_view)
        output_dir = paths.artifacts_path / "module3" / "selected" / str(slot)
        return SourceResolution(
            source_view=source_view,
            source_experiment=source_experiment,
            source_artifacts_path=source_artifacts_path,
            source_outcome_matrix_path=source_outcome_matrix_path,
            output_dir=output_dir,
            mode="slot",
            slot=str(slot),
        )

    if not view or not experiment:
        raise ValueError("Provide either --slot or both --view and --experiment.")

    source_outcome_matrix_path = resolve_outcome_matrix_path(paths.processed_path, str(view))
    return SourceResolution(
        source_view=str(view),
        source_experiment=str(experiment),
        source_artifacts_path=paths.artifacts_path / "module2" / str(view) / str(experiment),
        source_outcome_matrix_path=source_outcome_matrix_path,
        output_dir=paths.artifacts_path / "module3" / str(view) / str(experiment),
        mode="direct",
    )


def read_analysis_inputs(
    outcome_matrix_path: Path,
    membership_matrix_path: Path,
    threshold: float,
    required_outcome_cols: set[str] | None = None,
) -> AnalysisArtifacts:
    if not outcome_matrix_path.exists():
        raise FileNotFoundError(f"Outcome matrix not found: {outcome_matrix_path}")
    if not membership_matrix_path.exists():
        raise FileNotFoundError(f"Membership matrix not found: {membership_matrix_path}")

    outcome_df = read_parquet_with_compat_hint(outcome_matrix_path)
    membership_df = read_parquet_with_compat_hint(membership_matrix_path)

    required_outcome_cols = set(required_outcome_cols or {PRIMARY_OUTCOME_GLYCEMIC_CV})
    required_cols = required_outcome_cols | {"hba1c", "diabetes_stage"}
    missing_outcome = sorted(required_cols - set(outcome_df.columns))
    if missing_outcome:
        raise ValueError(f"Outcome matrix missing required columns: {missing_outcome}")

    probability_columns = _probability_columns(membership_df.columns.tolist())
    if len(probability_columns) < 2:
        raise ValueError("Membership matrix must contain at least two cluster probability columns named like pi_1, pi_2.")

    validate_probability_matrix(membership_df.loc[:, probability_columns])
    aligned_outcome = align_outcome_and_membership(outcome_df, membership_df)

    glycemic_cv = pd.to_numeric(aligned_outcome[PRIMARY_OUTCOME_GLYCEMIC_CV], errors="raise")
    hba1c = pd.to_numeric(aligned_outcome["hba1c"], errors="raise")
    diabetes_stage_raw = aligned_outcome["diabetes_stage"].copy()
    diabetes_stage = normalize_diabetes_stage_series(diabetes_stage_raw)

    optional_numeric_cols = {
        "mean_glucose",
        "time_in_range",
        SECONDARY_OUTCOME_TIME_BELOW_70,
        OPTIONAL_OUTCOME_TIME_BELOW_54,
    }
    numeric_outcomes: dict[str, pd.Series] = {}
    for col in optional_numeric_cols:
        if col in aligned_outcome.columns:
            numeric_outcomes[col] = pd.to_numeric(aligned_outcome[col], errors="raise")

    if glycemic_cv.isna().any() or hba1c.isna().any() or diabetes_stage.isna().any():
        raise ValueError("Outcome matrix contains missing values in glycemic_cv, hba1c, or diabetes_stage.")

    for col in (SECONDARY_OUTCOME_TIME_BELOW_70, OPTIONAL_OUTCOME_TIME_BELOW_54, "time_in_range"):
        if col not in numeric_outcomes:
            continue
        series = numeric_outcomes[col]
        if series.isna().any():
            raise ValueError(f"Outcome matrix contains missing values in {col}.")
        if ((series < 0) | (series > 1)).any():
            raise ValueError(f"{col} must lie within [0, 1].")

    stable = (glycemic_cv < threshold).astype(int)
    unstable = 1 - stable

    hba1c_mean = float(hba1c.mean())
    hba1c_std = float(hba1c.std(ddof=0))
    if not np.isfinite(hba1c_std) or hba1c_std <= 0:
        raise ValueError("HbA1c has zero or non-finite variance; Module 3 requires variability in hba1c.")

    diabetes_stage_int = diabetes_stage.astype(int)
    observed_stages = sorted(int(v) for v in diabetes_stage_int.unique().tolist())
    reference_stage = 0 if 0 in observed_stages else int(min(observed_stages))

    reference_cluster = str(membership_df.loc[:, probability_columns].sum(axis=0).idxmax())
    predictor_probability_columns = [c for c in probability_columns if c != reference_cluster]

    top_cluster = membership_df.loc[:, probability_columns].idxmax(axis=1)
    max_membership = membership_df.loc[:, probability_columns].max(axis=1)
    second_membership = membership_df.loc[:, probability_columns].apply(_second_largest_probability, axis=1)
    membership_margin = max_membership - second_membership

    analysis_df = aligned_outcome.copy()
    analysis_df[PRIMARY_OUTCOME_GLYCEMIC_CV] = glycemic_cv
    analysis_df["hba1c"] = hba1c
    analysis_df[STABLE_LABEL] = stable
    analysis_df[UNSTABLE_LABEL] = unstable
    analysis_df[PREDICTOR_HBA1C_Z] = (hba1c - hba1c_mean) / hba1c_std
    analysis_df[DIABETES_STAGE_RAW_COL] = diabetes_stage_raw.astype(str)
    analysis_df["diabetes_stage"] = diabetes_stage_int
    analysis_df[TOP_CLUSTER_COL] = top_cluster
    analysis_df[MAX_MEMBERSHIP_COL] = max_membership
    analysis_df[MEMBERSHIP_MARGIN_COL] = membership_margin
    analysis_df.loc[:, probability_columns] = membership_df.loc[:, probability_columns]

    for col, series in numeric_outcomes.items():
        analysis_df[col] = series

    for stage in observed_stages:
        if stage == reference_stage:
            continue
        analysis_df[f"stage_{stage}"] = (analysis_df["diabetes_stage"] == stage).astype(float)

    diagnostics_df = build_diagnostics_table(
        analysis_df=analysis_df,
        probability_columns=probability_columns,
        reference_cluster=reference_cluster,
        reference_stage=reference_stage,
    )

    return AnalysisArtifacts(
        analysis_df=analysis_df,
        diagnostics_df=diagnostics_df,
        probability_columns=probability_columns,
        predictor_probability_columns=predictor_probability_columns,
        reference_cluster=reference_cluster,
        reference_stage=reference_stage,
        observed_stages=observed_stages,
        hba1c_mean=hba1c_mean,
        hba1c_std=hba1c_std,
        threshold=threshold,
    )


def _probability_columns(columns: list[str]) -> list[str]:
    prefixed = [c for c in columns if c.startswith("pi_")]
    return sorted(prefixed, key=_cluster_sort_key)


def _cluster_sort_key(col: str) -> tuple[int, str]:
    suffix = col.split("_", 1)[1] if "_" in col else col
    try:
        return (0, f"{int(suffix):06d}")
    except ValueError:
        return (1, suffix)


def _second_largest_probability(row: pd.Series) -> float:
    values = np.sort(row.to_numpy(dtype=float))
    return float(values[-2]) if len(values) > 1 else 0.0


def normalize_diabetes_stage_series(stage_series: pd.Series) -> pd.Series:
    normalized = stage_series.map(_normalize_single_diabetes_stage)
    return pd.Series(normalized, index=stage_series.index, name="diabetes_stage", dtype="int64")


def _normalize_single_diabetes_stage(value: Any) -> int:
    if pd.isna(value):
        raise ValueError("diabetes_stage contains missing values.")

    try:
        numeric = float(value)
        if np.isfinite(numeric) and numeric.is_integer():
            integer = int(numeric)
            if integer in STAGE_LABEL_ALIASES:
                return integer
    except (TypeError, ValueError):
        pass

    normalized = _normalize_stage_label(value)
    for stage_code, aliases in STAGE_LABEL_ALIASES.items():
        if normalized in aliases:
            return stage_code

    if "prediab" in normalized or "lifestyle" in normalized:
        return 1
    if "non_insulin" in normalized or ("oral" in normalized and "insulin" not in normalized):
        return 2
    if "insulin" in normalized:
        return 3
    if normalized in {"control", "healthy"} or ("diabet" in normalized and normalized.startswith("no_")):
        return 0

    allowed_examples = [
        "0/no_diabetes/control",
        "1/prediabetes/lifestyle_controlled",
        "2/non_insulin_dependent/oral_non_insulin_injectable",
        "3/insulin_dependent/insulin_controlled",
    ]
    raise ValueError(
        "Unsupported diabetes_stage value "
        f"{value!r}. Update the stage normalizer if your study_group labels differ. "
        f"Supported examples: {allowed_examples}"
    )


def _normalize_stage_label(value: Any) -> str:
    text = str(value).strip().lower()
    for old, new in (
        ("&", " and "),
        ("/", "_"),
        ("-", "_"),
        (" ", "_"),
        ("(", ""),
        (")", ""),
        (",", "_"),
    ):
        text = text.replace(old, new)
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def validate_probability_matrix(prob_df: pd.DataFrame) -> None:
    non_numeric = [c for c in prob_df.columns if not pd.api.types.is_numeric_dtype(prob_df[c])]
    if non_numeric:
        raise ValueError(f"Membership matrix contains non-numeric probability columns: {non_numeric}")

    values = prob_df.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Membership matrix contains non-finite values.")

    if (values < -PROBABILITY_TOLERANCE).any() or (values > 1 + PROBABILITY_TOLERANCE).any():
        raise ValueError("Membership probabilities must lie within [0, 1].")

    row_sums = values.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=PROBABILITY_TOLERANCE):
        max_deviation = float(np.abs(row_sums - 1.0).max())
        raise ValueError(
            f"Membership probabilities must sum to 1 for every participant; max deviation was {max_deviation:.6g}."
        )


def align_outcome_and_membership(outcome_df: pd.DataFrame, membership_df: pd.DataFrame) -> pd.DataFrame:
    outcome_ids = set(outcome_df.index.tolist())
    membership_ids = set(membership_df.index.tolist())
    if outcome_ids != membership_ids:
        only_outcome = sorted(outcome_ids - membership_ids)[:5]
        only_membership = sorted(membership_ids - outcome_ids)[:5]
        raise ValueError(
            "Outcome matrix and membership matrix must contain the same participant IDs. "
            f"Only in outcome: {only_outcome}; only in membership: {only_membership}"
        )
    return outcome_df.loc[membership_df.index].copy()


def build_diagnostics_table(
    analysis_df: pd.DataFrame,
    probability_columns: list[str],
    reference_cluster: str,
    reference_stage: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add_row(section: str, group: str, metric: str, value: Any) -> None:
        numeric_value = None
        if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value)):
            numeric_value = float(value)
        rows.append(
            {
                "section": section,
                "group": group,
                "metric": metric,
                "value": str(value),
                "value_numeric": numeric_value,
            }
        )

    add_row("cohort", "all", "n_participants", int(len(analysis_df)))
    add_row("model_reference", "cluster", "reference_cluster", reference_cluster)
    add_row("model_reference", "diabetes_stage", "reference_stage", int(reference_stage))

    stage_counts = analysis_df["diabetes_stage"].value_counts().sort_index()
    for stage, count in stage_counts.items():
        add_row("diabetes_stage", str(int(stage)), "n_participants", int(count))

    stable_counts = analysis_df[STABLE_LABEL].value_counts().sort_index()
    add_row("stable_outcome", "stable", "n_participants", int(stable_counts.get(1, 0)))
    add_row("stable_outcome", "unstable", "n_participants", int(stable_counts.get(0, 0)))
    add_row("stable_outcome", "all", "stable_rate", float(analysis_df[STABLE_LABEL].mean()))

    for outcome_name in [
        PRIMARY_OUTCOME_GLYCEMIC_CV,
        "mean_glucose",
        "time_in_range",
        SECONDARY_OUTCOME_TIME_BELOW_70,
        OPTIONAL_OUTCOME_TIME_BELOW_54,
    ]:
        if outcome_name not in analysis_df.columns:
            continue
        outcome = pd.to_numeric(analysis_df[outcome_name], errors="coerce")
        add_row("outcome_distribution", outcome_name, "mean", float(outcome.mean()))
        add_row("outcome_distribution", outcome_name, "median", float(outcome.median()))
        add_row("outcome_distribution", outcome_name, "p25", float(outcome.quantile(0.25)))
        add_row("outcome_distribution", outcome_name, "p75", float(outcome.quantile(0.75)))
        add_row("outcome_distribution", outcome_name, "min", float(outcome.min()))
        add_row("outcome_distribution", outcome_name, "max", float(outcome.max()))
        if outcome_name.startswith("time_"):
            add_row("outcome_distribution", outcome_name, "zero_rate", float((outcome == 0).mean()))

    top_cluster_counts = analysis_df[TOP_CLUSTER_COL].value_counts().sort_index()
    for cluster, count in top_cluster_counts.items():
        add_row("top_cluster", str(cluster), "n_participants", int(count))

    membership = analysis_df.loc[:, probability_columns]
    for cluster in probability_columns:
        add_row("cluster_membership", cluster, "mean_membership", float(membership[cluster].mean()))
        add_row("cluster_membership", cluster, "total_membership_mass", float(membership[cluster].sum()))

    borderline_max = int((analysis_df[MAX_MEMBERSHIP_COL] < BORDERLINE_MAX_THRESHOLD).sum())
    borderline_margin = int((analysis_df[MEMBERSHIP_MARGIN_COL] < BORDERLINE_MARGIN_THRESHOLD).sum())
    add_row("cluster_uncertainty", "all", "max_membership_lt_0_60", borderline_max)
    add_row("cluster_uncertainty", "all", "membership_margin_lt_0_20", borderline_margin)

    return pd.DataFrame(rows)


def import_bayesian_dependencies():
    if "PYTENSOR_FLAGS" not in os.environ:
        pytensor_cache_dir = Path("/tmp/pytensor").resolve()
        pytensor_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["PYTENSOR_FLAGS"] = f"base_compiledir={pytensor_cache_dir}"
    try:
        import arviz as az
        import pymc as pm
    except ImportError as exc:
        raise ImportError(
            "Module 3 requires 'pymc' and 'arviz' in the active environment. "
            "Install the environment defined in environment.yml before running this pipeline."
        ) from exc
    return pm, az


def build_design_matrix(artifacts: AnalysisArtifacts) -> tuple[pd.DataFrame, list[str]]:
    stage_columns = [f"stage_{stage}" for stage in artifacts.observed_stages if stage != artifacts.reference_stage]
    predictor_columns = [PREDICTOR_HBA1C_Z] + stage_columns + artifacts.predictor_probability_columns
    design_matrix = artifacts.analysis_df.loc[:, predictor_columns].astype(float)
    return design_matrix, predictor_columns


def validate_threshold_outcome(artifacts: AnalysisArtifacts) -> None:
    stable = artifacts.analysis_df[STABLE_LABEL]
    if stable.nunique() < 2:
        raise ValueError(
            "Stable-glycemia outcome has no variation after thresholding; both stable and unstable participants are required."
        )


def fit_threshold_model(
    cfg: dict[str, Any],
    artifacts: AnalysisArtifacts,
) -> ModelFitArtifacts:
    validate_threshold_outcome(artifacts)

    pm, az = import_bayesian_dependencies()
    design_matrix, predictor_columns = build_design_matrix(artifacts)
    y = artifacts.analysis_df[STABLE_LABEL].astype(int).to_numpy()

    random_seed = int(cfg.get("module2", {}).get("random_seed", DEFAULT_RANDOM_SEED))
    sampling_cfg = cfg.get("module3", {}).get("sampling", {})
    coords = {
        "participant": artifacts.analysis_df.index.astype(str).tolist(),
        "feature": predictor_columns,
    }

    with pm.Model(coords=coords) as model:
        x_data = pm.Data("X", design_matrix.to_numpy(dtype=float), dims=("participant", "feature"))

        intercept = pm.Normal("intercept", mu=0.0, sigma=1.5)
        beta = pm.Normal("beta", mu=0.0, sigma=1.0, dims="feature")
        linear = intercept + pm.math.dot(x_data, beta)
        p_stable = pm.Deterministic("p_stable", pm.math.sigmoid(linear), dims="participant")
        pm.Bernoulli("stable_obs", p=p_stable, observed=y, dims="participant")

        idata = pm.sample(
            draws=int(sampling_cfg.get("draws", 2000)),
            tune=int(sampling_cfg.get("tune", 1000)),
            chains=int(sampling_cfg.get("chains", 4)),
            target_accept=float(sampling_cfg.get("target_accept", 0.9)),
            random_seed=random_seed,
            progressbar=False,
            cores=1,
            return_inferencedata=True,
        )
        idata.extend(
            pm.sample_posterior_predictive(
                idata,
                var_names=["stable_obs"],
                predictions=False,
                progressbar=False,
                random_seed=random_seed,
            )
        )

    summary_df = az.summary(idata, var_names=["intercept", "beta"], round_to=None).reset_index()
    summary_df = summary_df.rename(columns={"index": "term"})

    sampling_diagnostics = build_sampling_diagnostics(idata=idata, summary_df=summary_df)
    posterior_predictive = build_threshold_posterior_predictive_summary(idata=idata, observed=y)
    participant_predictions = build_threshold_participant_predictions(artifacts, idata)
    cluster_stage_predictions = build_threshold_cluster_stage_predictions(artifacts, idata)

    return ModelFitArtifacts(
        idata=idata,
        model_summary_df=summary_df,
        participant_predictions=participant_predictions,
        cluster_stage_predictions=cluster_stage_predictions,
        sampling_diagnostics=sampling_diagnostics,
        posterior_predictive=posterior_predictive,
        run_metadata={
            "target_probability": "P(CV < threshold | hba1c, diabetes_stage, cluster_probability_profile)",
        },
    )


def prepare_continuous_outcomes(artifacts: AnalysisArtifacts, epsilon: float) -> ContinuousOutcomeData:
    glycemic_cv = pd.to_numeric(artifacts.analysis_df[PRIMARY_OUTCOME_GLYCEMIC_CV], errors="raise").to_numpy(dtype=float)
    time_below_70 = pd.to_numeric(
        artifacts.analysis_df[SECONDARY_OUTCOME_TIME_BELOW_70], errors="raise"
    ).to_numpy(dtype=float)

    if not np.isfinite(glycemic_cv).all():
        raise ValueError("glycemic_cv contains non-finite values.")
    if not np.isfinite(time_below_70).all():
        raise ValueError("time_below_70 contains non-finite values.")
    if ((time_below_70 < 0) | (time_below_70 > 1)).any():
        raise ValueError("time_below_70 must lie within [0, 1].")

    glycemic_cv_mean = float(glycemic_cv.mean())
    glycemic_cv_std = float(glycemic_cv.std(ddof=0))
    if not np.isfinite(glycemic_cv_std) or glycemic_cv_std <= 0:
        raise ValueError("glycemic_cv has zero or non-finite variance; continuous modeling requires variability.")

    clipped_tbr = clip_proportion(time_below_70, epsilon)
    tbr_logit = logit(clipped_tbr)
    tbr_logit_mean = float(tbr_logit.mean())
    tbr_logit_std = float(tbr_logit.std(ddof=0))
    if not np.isfinite(tbr_logit_std) or tbr_logit_std <= 0:
        raise ValueError("time_below_70 has zero or non-finite variance after transformation.")

    return ContinuousOutcomeData(
        glycemic_cv_z=(glycemic_cv - glycemic_cv_mean) / glycemic_cv_std,
        glycemic_cv_mean=glycemic_cv_mean,
        glycemic_cv_std=glycemic_cv_std,
        time_below_70_logit_z=(tbr_logit - tbr_logit_mean) / tbr_logit_std,
        time_below_70_logit_mean=tbr_logit_mean,
        time_below_70_logit_std=tbr_logit_std,
        proportion_epsilon=epsilon,
    )


def fit_continuous_dual_model(
    cfg: dict[str, Any],
    artifacts: AnalysisArtifacts,
) -> ModelFitArtifacts:
    pm, az = import_bayesian_dependencies()
    design_matrix, predictor_columns = build_design_matrix(artifacts)
    module3_cfg = cfg.get("module3", {})
    epsilon = float(module3_cfg.get("proportion_epsilon", DEFAULT_PROPORTION_EPSILON))
    outcome_data = prepare_continuous_outcomes(artifacts, epsilon)

    random_seed = int(cfg.get("module2", {}).get("random_seed", DEFAULT_RANDOM_SEED))
    sampling_cfg = module3_cfg.get("sampling", {})
    coords = {
        "participant": artifacts.analysis_df.index.astype(str).tolist(),
        "feature": predictor_columns,
    }

    with pm.Model(coords=coords) as model:
        x_data = pm.Data("X", design_matrix.to_numpy(dtype=float), dims=("participant", "feature"))

        glycemic_cv_intercept = pm.Normal("glycemic_cv_intercept", mu=0.0, sigma=1.5)
        glycemic_cv_beta = pm.Normal("glycemic_cv_beta", mu=0.0, sigma=1.0, dims="feature")
        glycemic_cv_sigma = pm.HalfNormal("glycemic_cv_sigma", sigma=1.0)
        glycemic_cv_mu_z = glycemic_cv_intercept + pm.math.dot(x_data, glycemic_cv_beta)
        pm.Normal(
            "glycemic_cv_z_obs",
            mu=glycemic_cv_mu_z,
            sigma=glycemic_cv_sigma,
            observed=outcome_data.glycemic_cv_z,
            dims="participant",
        )
        pm.Deterministic(
            "glycemic_cv_pred",
            glycemic_cv_mu_z * outcome_data.glycemic_cv_std + outcome_data.glycemic_cv_mean,
            dims="participant",
        )

        time_below_70_intercept = pm.Normal("time_below_70_intercept", mu=0.0, sigma=1.5)
        time_below_70_beta = pm.Normal("time_below_70_beta", mu=0.0, sigma=1.0, dims="feature")
        time_below_70_sigma = pm.HalfNormal("time_below_70_sigma", sigma=1.0)
        time_below_70_mu_z = time_below_70_intercept + pm.math.dot(x_data, time_below_70_beta)
        pm.Normal(
            "time_below_70_logit_z_obs",
            mu=time_below_70_mu_z,
            sigma=time_below_70_sigma,
            observed=outcome_data.time_below_70_logit_z,
            dims="participant",
        )
        time_below_70_logit_pred = pm.Deterministic(
            "time_below_70_logit_pred",
            time_below_70_mu_z * outcome_data.time_below_70_logit_std + outcome_data.time_below_70_logit_mean,
            dims="participant",
        )
        pm.Deterministic(
            "time_below_70_pred",
            pm.math.sigmoid(time_below_70_logit_pred),
            dims="participant",
        )

        idata = pm.sample(
            draws=int(sampling_cfg.get("draws", 2000)),
            tune=int(sampling_cfg.get("tune", 1000)),
            chains=int(sampling_cfg.get("chains", 4)),
            target_accept=float(sampling_cfg.get("target_accept", 0.9)),
            random_seed=random_seed,
            progressbar=False,
            cores=1,
            return_inferencedata=True,
        )
        idata.extend(
            pm.sample_posterior_predictive(
                idata,
                var_names=["glycemic_cv_z_obs", "time_below_70_logit_z_obs"],
                predictions=False,
                progressbar=False,
                random_seed=random_seed,
            )
        )

    summary_df = az.summary(
        idata,
        var_names=[
            "glycemic_cv_intercept",
            "glycemic_cv_beta",
            "glycemic_cv_sigma",
            "time_below_70_intercept",
            "time_below_70_beta",
            "time_below_70_sigma",
        ],
        round_to=None,
    ).reset_index()
    summary_df = summary_df.rename(columns={"index": "term"})

    sampling_diagnostics = build_sampling_diagnostics(idata=idata, summary_df=summary_df)
    posterior_predictive = build_continuous_posterior_predictive_summary(artifacts, idata, outcome_data)
    participant_predictions = build_continuous_participant_predictions(artifacts, idata)
    cluster_stage_predictions = build_continuous_cluster_stage_predictions(artifacts, idata, outcome_data)

    return ModelFitArtifacts(
        idata=idata,
        model_summary_df=summary_df,
        participant_predictions=participant_predictions,
        cluster_stage_predictions=cluster_stage_predictions,
        sampling_diagnostics=sampling_diagnostics,
        posterior_predictive=posterior_predictive,
        run_metadata={
            "target_estimands": [
                "E(glycemic_cv | hba1c, diabetes_stage, cluster_probability_profile)",
                "E(time_below_70 | hba1c, diabetes_stage, cluster_probability_profile)",
            ],
            "transformations": {
                PRIMARY_OUTCOME_GLYCEMIC_CV: {
                    "latent_scale": "zscore",
                    "mean": outcome_data.glycemic_cv_mean,
                    "std": outcome_data.glycemic_cv_std,
                },
                SECONDARY_OUTCOME_TIME_BELOW_70: {
                    "latent_scale": "logit_then_zscore",
                    "epsilon": outcome_data.proportion_epsilon,
                    "logit_mean": outcome_data.time_below_70_logit_mean,
                    "logit_std": outcome_data.time_below_70_logit_std,
                },
            },
        },
    )


def build_sampling_diagnostics(idata: Any, summary_df: pd.DataFrame) -> dict[str, Any]:
    divergences = int(np.asarray(idata.sample_stats["diverging"]).sum())
    max_rhat = float(summary_df["r_hat"].max()) if "r_hat" in summary_df.columns else float("nan")
    min_ess_bulk = float(summary_df["ess_bulk"].min()) if "ess_bulk" in summary_df.columns else float("nan")
    diagnostics_ok = (
        divergences == 0
        and (not np.isfinite(max_rhat) or max_rhat <= R_HAT_THRESHOLD)
        and (not np.isfinite(min_ess_bulk) or min_ess_bulk >= ESS_BULK_THRESHOLD)
    )
    return {
        "divergences": divergences,
        "max_rhat": max_rhat,
        "min_ess_bulk": min_ess_bulk,
        "rhat_threshold": R_HAT_THRESHOLD,
        "ess_bulk_threshold": ESS_BULK_THRESHOLD,
        "diagnostics_ok": bool(diagnostics_ok),
    }


def build_threshold_posterior_predictive_summary(idata: Any, observed: np.ndarray) -> dict[str, Any]:
    ppc = np.asarray(idata.posterior_predictive["stable_obs"])
    if ppc.ndim != 3:
        raise ValueError(f"Unexpected posterior predictive shape for stable_obs: {ppc.shape}")
    stable_rate_samples = ppc.mean(axis=2).reshape(-1)
    stable_count_samples = ppc.sum(axis=2).reshape(-1)
    return {
        "observed_stable_rate": float(observed.mean()),
        "observed_stable_count": int(observed.sum()),
        "posterior_predictive_stable_rate_mean": float(stable_rate_samples.mean()),
        "posterior_predictive_stable_rate_hdi_low": float(np.quantile(stable_rate_samples, 0.025)),
        "posterior_predictive_stable_rate_hdi_high": float(np.quantile(stable_rate_samples, 0.975)),
        "posterior_predictive_stable_count_mean": float(stable_count_samples.mean()),
        "posterior_predictive_stable_count_hdi_low": float(np.quantile(stable_count_samples, 0.025)),
        "posterior_predictive_stable_count_hdi_high": float(np.quantile(stable_count_samples, 0.975)),
    }


def build_continuous_posterior_predictive_summary(
    artifacts: AnalysisArtifacts,
    idata: Any,
    outcome_data: ContinuousOutcomeData,
) -> dict[str, Any]:
    glycemic_cv_ppc_z = np.asarray(idata.posterior_predictive["glycemic_cv_z_obs"])
    time_below_70_ppc_z = np.asarray(idata.posterior_predictive["time_below_70_logit_z_obs"])
    if glycemic_cv_ppc_z.ndim != 3 or time_below_70_ppc_z.ndim != 3:
        raise ValueError(
            "Unexpected posterior predictive shape for continuous_dual outputs: "
            f"{glycemic_cv_ppc_z.shape}, {time_below_70_ppc_z.shape}"
        )

    glycemic_cv_ppc = glycemic_cv_ppc_z * outcome_data.glycemic_cv_std + outcome_data.glycemic_cv_mean
    time_below_70_ppc = inverse_logit(
        time_below_70_ppc_z * outcome_data.time_below_70_logit_std + outcome_data.time_below_70_logit_mean
    )

    observed_cv = pd.to_numeric(artifacts.analysis_df[PRIMARY_OUTCOME_GLYCEMIC_CV], errors="raise").to_numpy(dtype=float)
    observed_tbr = pd.to_numeric(artifacts.analysis_df[SECONDARY_OUTCOME_TIME_BELOW_70], errors="raise").to_numpy(dtype=float)

    cv_mean_samples = glycemic_cv_ppc.mean(axis=2).reshape(-1)
    tbr_mean_samples = time_below_70_ppc.mean(axis=2).reshape(-1)

    return {
        PRIMARY_OUTCOME_GLYCEMIC_CV: {
            "observed_mean": float(observed_cv.mean()),
            "observed_median": float(np.median(observed_cv)),
            "posterior_predictive_mean_mean": float(cv_mean_samples.mean()),
            "posterior_predictive_mean_hdi_low": float(np.quantile(cv_mean_samples, 0.025)),
            "posterior_predictive_mean_hdi_high": float(np.quantile(cv_mean_samples, 0.975)),
        },
        SECONDARY_OUTCOME_TIME_BELOW_70: {
            "observed_mean": float(observed_tbr.mean()),
            "observed_median": float(np.median(observed_tbr)),
            "posterior_predictive_mean_mean": float(tbr_mean_samples.mean()),
            "posterior_predictive_mean_hdi_low": float(np.quantile(tbr_mean_samples, 0.025)),
            "posterior_predictive_mean_hdi_high": float(np.quantile(tbr_mean_samples, 0.975)),
        },
    }


def build_threshold_participant_predictions(artifacts: AnalysisArtifacts, idata: Any) -> pd.DataFrame:
    p_samples = np.asarray(idata.posterior["p_stable"])
    if p_samples.ndim != 3:
        raise ValueError(f"Unexpected posterior shape for p_stable: {p_samples.shape}")
    draws_by_participant = p_samples.reshape(-1, p_samples.shape[-1]).T

    stable_mean = draws_by_participant.mean(axis=1)
    stable_low = np.quantile(draws_by_participant, 0.025, axis=1)
    stable_high = np.quantile(draws_by_participant, 0.975, axis=1)

    pred_df = pd.DataFrame(index=artifacts.analysis_df.index)
    pred_df[STABLE_LABEL] = artifacts.analysis_df[STABLE_LABEL].astype(int)
    pred_df["p_stable_mean"] = stable_mean
    pred_df["p_stable_hdi_low"] = stable_low
    pred_df["p_stable_hdi_high"] = stable_high
    pred_df["p_unstable_mean"] = 1.0 - stable_mean
    pred_df["p_unstable_hdi_low"] = 1.0 - stable_high
    pred_df["p_unstable_hdi_high"] = 1.0 - stable_low
    return append_prediction_context(pred_df, artifacts)


def build_continuous_participant_predictions(artifacts: AnalysisArtifacts, idata: Any) -> pd.DataFrame:
    glycemic_cv_samples = _draws_by_participant(np.asarray(idata.posterior["glycemic_cv_pred"]), "glycemic_cv_pred")
    time_below_70_samples = _draws_by_participant(np.asarray(idata.posterior["time_below_70_pred"]), "time_below_70_pred")

    pred_df = pd.DataFrame(index=artifacts.analysis_df.index)
    pred_df[PRIMARY_OUTCOME_GLYCEMIC_CV] = artifacts.analysis_df[PRIMARY_OUTCOME_GLYCEMIC_CV].astype(float)
    pred_df["glycemic_cv_pred_mean"] = glycemic_cv_samples.mean(axis=1)
    pred_df["glycemic_cv_pred_hdi_low"] = np.quantile(glycemic_cv_samples, 0.025, axis=1)
    pred_df["glycemic_cv_pred_hdi_high"] = np.quantile(glycemic_cv_samples, 0.975, axis=1)
    pred_df[SECONDARY_OUTCOME_TIME_BELOW_70] = artifacts.analysis_df[SECONDARY_OUTCOME_TIME_BELOW_70].astype(float)
    pred_df["time_below_70_pred_mean"] = time_below_70_samples.mean(axis=1)
    pred_df["time_below_70_pred_hdi_low"] = np.quantile(time_below_70_samples, 0.025, axis=1)
    pred_df["time_below_70_pred_hdi_high"] = np.quantile(time_below_70_samples, 0.975, axis=1)
    if OPTIONAL_OUTCOME_TIME_BELOW_54 in artifacts.analysis_df.columns:
        pred_df[OPTIONAL_OUTCOME_TIME_BELOW_54] = artifacts.analysis_df[OPTIONAL_OUTCOME_TIME_BELOW_54].astype(float)
    return append_prediction_context(pred_df, artifacts)


def append_prediction_context(pred_df: pd.DataFrame, artifacts: AnalysisArtifacts) -> pd.DataFrame:
    pred_df[STABLE_LABEL] = artifacts.analysis_df[STABLE_LABEL].astype(int)
    pred_df["diabetes_stage"] = artifacts.analysis_df["diabetes_stage"]
    pred_df[DIABETES_STAGE_RAW_COL] = artifacts.analysis_df[DIABETES_STAGE_RAW_COL]
    pred_df["hba1c"] = artifacts.analysis_df["hba1c"]
    pred_df[TOP_CLUSTER_COL] = artifacts.analysis_df[TOP_CLUSTER_COL]
    pred_df[MAX_MEMBERSHIP_COL] = artifacts.analysis_df[MAX_MEMBERSHIP_COL]
    pred_df[MEMBERSHIP_MARGIN_COL] = artifacts.analysis_df[MEMBERSHIP_MARGIN_COL]
    return pred_df


def build_threshold_cluster_stage_predictions(
    artifacts: AnalysisArtifacts,
    idata: Any,
) -> pd.DataFrame:
    beta_samples = np.asarray(idata.posterior["beta"])
    intercept_samples = np.asarray(idata.posterior["intercept"]).reshape(-1)
    beta_samples = beta_samples.reshape(-1, beta_samples.shape[-1])

    _, predictor_columns = build_design_matrix(artifacts)
    rows: list[dict[str, Any]] = []
    for stage, cluster, x in iter_cluster_stage_vectors(artifacts, predictor_columns):
        linear_samples = intercept_samples + beta_samples.dot(x)
        p_samples = inverse_logit(linear_samples)
        rows.append(
            {
                "diabetes_stage": int(stage),
                "cluster_profile": cluster,
                "hba1c": artifacts.hba1c_mean,
                "p_stable_mean": float(p_samples.mean()),
                "p_stable_hdi_low": float(np.quantile(p_samples, 0.025)),
                "p_stable_hdi_high": float(np.quantile(p_samples, 0.975)),
                "p_unstable_mean": float(1.0 - p_samples.mean()),
                "p_unstable_hdi_low": float(1.0 - np.quantile(p_samples, 0.975)),
                "p_unstable_hdi_high": float(1.0 - np.quantile(p_samples, 0.025)),
            }
        )

    return pd.DataFrame(rows)


def build_continuous_cluster_stage_predictions(
    artifacts: AnalysisArtifacts,
    idata: Any,
    outcome_data: ContinuousOutcomeData,
) -> pd.DataFrame:
    glycemic_cv_beta = np.asarray(idata.posterior["glycemic_cv_beta"]).reshape(-1, len(build_design_matrix(artifacts)[1]))
    glycemic_cv_intercept = np.asarray(idata.posterior["glycemic_cv_intercept"]).reshape(-1)
    time_below_70_beta = np.asarray(idata.posterior["time_below_70_beta"]).reshape(-1, len(build_design_matrix(artifacts)[1]))
    time_below_70_intercept = np.asarray(idata.posterior["time_below_70_intercept"]).reshape(-1)

    _, predictor_columns = build_design_matrix(artifacts)
    rows: list[dict[str, Any]] = []
    for stage, cluster, x in iter_cluster_stage_vectors(artifacts, predictor_columns):
        glycemic_cv_z_samples = glycemic_cv_intercept + glycemic_cv_beta.dot(x)
        glycemic_cv_samples = glycemic_cv_z_samples * outcome_data.glycemic_cv_std + outcome_data.glycemic_cv_mean

        time_below_70_logit_z_samples = time_below_70_intercept + time_below_70_beta.dot(x)
        time_below_70_samples = inverse_logit(
            time_below_70_logit_z_samples * outcome_data.time_below_70_logit_std
            + outcome_data.time_below_70_logit_mean
        )

        rows.append(
            {
                "diabetes_stage": int(stage),
                "cluster_profile": cluster,
                "hba1c": artifacts.hba1c_mean,
                "glycemic_cv_pred_mean": float(glycemic_cv_samples.mean()),
                "glycemic_cv_pred_hdi_low": float(np.quantile(glycemic_cv_samples, 0.025)),
                "glycemic_cv_pred_hdi_high": float(np.quantile(glycemic_cv_samples, 0.975)),
                "time_below_70_pred_mean": float(time_below_70_samples.mean()),
                "time_below_70_pred_hdi_low": float(np.quantile(time_below_70_samples, 0.025)),
                "time_below_70_pred_hdi_high": float(np.quantile(time_below_70_samples, 0.975)),
            }
        )

    return pd.DataFrame(rows)


def iter_cluster_stage_vectors(
    artifacts: AnalysisArtifacts,
    predictor_columns: list[str],
):
    mean_hba1c_z = 0.0
    for stage in artifacts.observed_stages:
        stage_vector = {name: 0.0 for name in predictor_columns}
        stage_vector[PREDICTOR_HBA1C_Z] = mean_hba1c_z
        if stage != artifacts.reference_stage:
            stage_vector[f"stage_{stage}"] = 1.0

        for cluster in artifacts.probability_columns:
            cluster_vector = dict(stage_vector)
            for prob_col in artifacts.predictor_probability_columns:
                cluster_vector[prob_col] = 0.0
            if cluster != artifacts.reference_cluster:
                cluster_vector[cluster] = 1.0

            x = np.array([cluster_vector[name] for name in predictor_columns], dtype=float)
            yield stage, cluster, x


def _draws_by_participant(samples: np.ndarray, name: str) -> np.ndarray:
    if samples.ndim != 3:
        raise ValueError(f"Unexpected posterior shape for {name}: {samples.shape}")
    return samples.reshape(-1, samples.shape[-1]).T


def clip_proportion(values: np.ndarray, epsilon: float) -> np.ndarray:
    if epsilon <= 0 or epsilon >= 0.5:
        raise ValueError("proportion_epsilon must lie in (0, 0.5).")
    return np.clip(values, epsilon, 1.0 - epsilon)


def logit(values: np.ndarray) -> np.ndarray:
    return np.log(values / (1.0 - values))


def inverse_logit(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def run_pipeline(
    cfg_path: Path,
    slot: str | None = None,
    view: str | None = None,
    experiment: str | None = None,
) -> Path:
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = load_config(cfg_path)
    module3_cfg = validate_module3_config(cfg)
    primary_model = str(module3_cfg.get("primary_model", CONTINUOUS_DUAL_MODEL))
    paths = resolve_paths(cfg)
    source = resolve_source(cfg, slot=slot, view=view, experiment=experiment)
    ensure_dir(source.output_dir)

    threshold = float(module3_cfg.get("cv_clinical_threshold", 36.0))
    membership_matrix_path = source.source_artifacts_path / "membership_matrix.parquet"

    artifacts = read_analysis_inputs(
        outcome_matrix_path=source.source_outcome_matrix_path,
        membership_matrix_path=membership_matrix_path,
        threshold=threshold,
        required_outcome_cols=required_outcome_columns(module3_cfg),
    )

    analysis_dataset_path = source.output_dir / "analysis_dataset.parquet"
    diagnostics_table_path = source.output_dir / "diagnostics_table.parquet"
    artifacts.analysis_df.to_parquet(analysis_dataset_path)
    artifacts.diagnostics_df.to_parquet(diagnostics_table_path, index=False)

    if primary_model == THRESHOLD_MODEL:
        fit_artifacts = fit_threshold_model(cfg, artifacts)
    else:
        fit_artifacts = fit_continuous_dual_model(cfg, artifacts)

    sampling_diagnostics_path = source.output_dir / "sampling_diagnostics.json"
    sampling_diagnostics_path.write_text(json.dumps(fit_artifacts.sampling_diagnostics, indent=2))
    if not fit_artifacts.sampling_diagnostics["diagnostics_ok"]:
        raise RuntimeError(
            "Bayesian sampling diagnostics failed. Inspect sampling_diagnostics.json and diagnostics_table.parquet."
        )

    model_summary_path = source.output_dir / "model_summary.parquet"
    participant_predictions_path = source.output_dir / "participant_predictions.parquet"
    cluster_stage_predictions_path = source.output_dir / "cluster_stage_predictions.parquet"
    posterior_predictive_path = source.output_dir / "posterior_predictive_summary.json"
    inference_data_path = source.output_dir / "inference_data.nc"
    run_summary_path = source.output_dir / "module3_run_summary.json"

    fit_artifacts.model_summary_df.to_parquet(model_summary_path, index=False)
    fit_artifacts.participant_predictions.to_parquet(participant_predictions_path)
    fit_artifacts.cluster_stage_predictions.to_parquet(cluster_stage_predictions_path, index=False)
    posterior_predictive_path.write_text(json.dumps(fit_artifacts.posterior_predictive, indent=2))

    _, az = import_bayesian_dependencies()
    az.to_netcdf(fit_artifacts.idata, inference_data_path)

    run_summary = {
        "status": "completed",
        "mode": source.mode,
        "slot": source.slot,
        "source_view": source.source_view,
        "source_experiment": source.source_experiment,
        "source_artifacts_path": str(source.source_artifacts_path),
        "source_outcome_matrix_path": str(source.source_outcome_matrix_path),
        "output_dir": str(source.output_dir),
        "primary_model": primary_model,
        "primary_outcome": str(module3_cfg.get("primary_outcome", PRIMARY_OUTCOME_GLYCEMIC_CV)),
        "secondary_outcomes": list(module3_cfg.get("secondary_outcomes", [])),
        "threshold": threshold,
        "reference_cluster": artifacts.reference_cluster,
        "reference_stage": artifacts.reference_stage,
        "hba1c_mean": artifacts.hba1c_mean,
        "hba1c_std": artifacts.hba1c_std,
        "n_participants": int(len(artifacts.analysis_df)),
        "probability_columns": artifacts.probability_columns,
        "predictor_probability_columns": artifacts.predictor_probability_columns,
        "sampling_diagnostics": fit_artifacts.sampling_diagnostics,
        "posterior_predictive_summary": fit_artifacts.posterior_predictive,
        "artifacts": {
            "analysis_dataset": str(analysis_dataset_path),
            "diagnostics_table": str(diagnostics_table_path),
            "model_summary": str(model_summary_path),
            "participant_predictions": str(participant_predictions_path),
            "cluster_stage_predictions": str(cluster_stage_predictions_path),
            "sampling_diagnostics": str(sampling_diagnostics_path),
            "posterior_predictive_summary": str(posterior_predictive_path),
            "inference_data": str(inference_data_path),
        },
    }
    run_summary.update(fit_artifacts.run_metadata)
    run_summary_path.write_text(json.dumps(run_summary, indent=2))
    return run_summary_path


def main() -> None:
    args = parse_args()
    run_summary_path = run_pipeline(
        cfg_path=Path(args.config),
        slot=args.slot,
        view=args.view,
        experiment=args.experiment,
    )
    print(f"Module 3 outputs written to {run_summary_path.parent}")


if __name__ == "__main__":
    main()
