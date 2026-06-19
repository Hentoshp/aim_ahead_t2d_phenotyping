import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.decomposition import PCA

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from module2_clustering.cluster_profiling import build_membership_matrix
from module2_clustering.dimensionality_reduction import PCAResult, run_pca, save_pca_artifacts
from module2_clustering.pipeline import run_pipeline
from module2_clustering.promote_solution import promote_solution
from module2_clustering.shap_importance import _normalize_shap_values


def _make_df(n_rows=200, n_features=5, seed=0):
    rng = np.random.default_rng(seed)
    # introduce correlation for meaningful PCA
    base = rng.normal(size=(n_rows, n_features))
    correlated = base @ np.triu(np.ones((n_features, n_features))) / n_features
    cols = [f"f{i}" for i in range(n_features)]
    return pd.DataFrame(correlated, columns=cols)


def test_run_pca_reaches_variance_threshold():
    df = _make_df()
    threshold = 0.9
    res = run_pca(df, variance_threshold=threshold, random_state=0)

    base_pca = PCA(random_state=0, svd_solver="full").fit(df.values)
    cumvar = np.cumsum(base_pca.explained_variance_ratio_)
    expected_n = int(np.searchsorted(cumvar, threshold) + 1)

    assert res.transformed.shape == (df.shape[0], expected_n)
    assert res.pca_model.n_components_ == expected_n
    assert res.explained_variance >= threshold - 1e-6


def test_run_pca_rejects_non_numeric():
    df = _make_df()
    df["bad"] = "text"
    with pytest.raises(ValueError):
        run_pca(df, variance_threshold=0.8)


def test_run_pca_rejects_nulls():
    df = _make_df()
    df.iloc[0, 0] = np.nan
    with pytest.raises(ValueError):
        run_pca(df, variance_threshold=0.8)


def test_save_pca_artifacts(tmp_path: Path):
    df = _make_df()
    res = run_pca(df, variance_threshold=0.85, random_state=42)

    model_path = save_pca_artifacts(res, tmp_path, feature_names=df.columns.tolist())

    summary_path = tmp_path / "pca_summary.json"
    assert model_path.exists()
    assert summary_path.exists()

    with open(summary_path) as f:
        summary = json.load(f)

    assert summary["n_components"] == res.pca_model.n_components_
    assert len(summary["variance_ratio"]) == res.pca_model.n_components_
    assert summary["n_features"] == df.shape[1]
    assert summary["feature_names"] == df.columns.tolist()
    assert summary["explained_variance"] == pytest.approx(res.explained_variance)


def test_run_pca_deterministic_with_seed():
    df = _make_df()
    res1 = run_pca(df, variance_threshold=0.9, random_state=7)
    res2 = run_pca(df, variance_threshold=0.9, random_state=7)

    assert np.allclose(res1.transformed, res2.transformed)
    assert np.allclose(res1.pca_model.components_, res2.pca_model.components_)


def test_normalize_shap_values_handles_3d_samples_features_classes():
    raw = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    normalized = _normalize_shap_values(raw, n_features=3)

    assert len(normalized) == 4
    assert all(arr.shape == (2, 3) for arr in normalized)
    assert np.allclose(normalized[0], raw[:, :, 0])


def test_normalize_shap_values_handles_3d_samples_classes_features():
    raw = np.arange(2 * 4 * 3, dtype=float).reshape(2, 4, 3)
    normalized = _normalize_shap_values(raw, n_features=3)

    assert len(normalized) == 4
    assert all(arr.shape == (2, 3) for arr in normalized)
    assert np.allclose(normalized[0], raw[:, 0, :])


def test_build_membership_matrix_supports_custom_output_name(tmp_path: Path):
    probs = np.array([[0.8, 0.2], [0.3, 0.7]])
    df = build_membership_matrix(["p1", "p2"], probs, artifacts_path=tmp_path, output_name="custom_membership.parquet")

    assert (tmp_path / "custom_membership.parquet").exists()
    assert list(df.columns) == ["pi_1", "pi_2"]
    assert np.allclose(df.sum(axis=1).to_numpy(), np.ones(len(df)))


def _write_module2_config(base_dir: Path) -> Path:
    cfg = {
        "data": {
            "processed_path": "${AIREADI_DATA_PATH}/processed/",
            "artifacts_path": "${AIREADI_DATA_PATH}/artifacts/",
        },
        "module1": {
            "clustering_views": {
                "default_view": "environment",
            },
        },
        "module2": {
            "k_range": [2],
            "covariance_types": ["diag"],
            "bootstrap_B": 1,
            "bootstrap_early_stop_threshold": 0.001,
            "pca_variance_threshold": 0.9,
            "pca_mode": "variance",
            "pca_n_components": None,
            "corr_prune": False,
            "corr_threshold": 0.9,
            "gmm_reg_covar": 0.001,
            "random_seed": 42,
            "artifacts": {
                "level": "standard",
                "compute_shap": False,
                "save_json_mirrors": False,
            },
        },
    }
    cfg_path = base_dir / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return cfg_path


def test_run_pipeline_writes_gmm_and_membership_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    processed_dir = tmp_path / "processed" / "clustering_views" / "environment"
    processed_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(11)
    cluster_a = rng.normal(loc=0.0, scale=0.25, size=(12, 4))
    cluster_b = rng.normal(loc=3.0, scale=0.25, size=(12, 4))
    matrix = pd.DataFrame(
        np.vstack([cluster_a, cluster_b]),
        index=[f"p{i:03d}" for i in range(24)],
        columns=["env_temp_median", "env_hum_median", "env_voc_median", "env_light_total_median"],
    )
    matrix.to_parquet(processed_dir / "clustering_matrix.parquet")

    cfg_path = _write_module2_config(tmp_path)
    monkeypatch.setenv("AIREADI_DATA_PATH", str(tmp_path))

    summary = run_pipeline(cfg_path=cfg_path, view="environment", experiment_name="smoke_test")

    artifacts_dir = tmp_path / "artifacts" / "module2" / "environment" / "smoke_test"
    assert summary["gmm_selection"]["best_k"] == 2
    assert (artifacts_dir / "gmm_model.joblib").exists()
    assert (artifacts_dir / "membership_matrix.parquet").exists()
    assert (artifacts_dir / "cluster_profiles.parquet").exists()


def test_promote_solution_rejects_stale_membership_vs_outcome(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    processed_view_dir = tmp_path / "processed" / "clustering_views" / "environment"
    artifacts_dir = tmp_path / "artifacts" / "module2" / "environment" / "stability_v1"
    processed_view_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    outcome_df = pd.DataFrame(
        {
            "glycemic_cv": [20.0, 25.0, 30.0],
            "mean_glucose": [110.0, 120.0, 130.0],
            "time_in_range": [0.9, 0.85, 0.8],
            "time_below_70": [0.0, 0.01, 0.02],
            "time_below_54": [0.0, 0.0, 0.0],
            "hba1c": [5.7, 6.4, 7.1],
            "diabetes_stage": [0, 1, 2],
        },
        index=["p1", "p2", "p3"],
    )
    membership_df = pd.DataFrame(
        {
            "pi_1": [0.8, 0.3],
            "pi_2": [0.2, 0.7],
        },
        index=["p1", "p2"],
    )
    outcome_df.to_parquet(processed_view_dir / "outcome_matrix.parquet")
    membership_df.to_parquet(artifacts_dir / "membership_matrix.parquet")
    pd.DataFrame({"cluster": [0], "feature": ["env_temp_median"], "centroid_value": [1.0]}).to_parquet(
        artifacts_dir / "cluster_profiles.parquet"
    )
    pd.DataFrame({"k": [2], "covariance_type": ["diag"], "bic": [1.0], "aic": [1.0], "log_likelihood": [-1.0]}).to_csv(
        artifacts_dir / "gmm_grid_search.csv",
        index=False,
    )
    joblib.dump({"kind": "pca"}, artifacts_dir / "pca_model.joblib")
    joblib.dump({"kind": "gmm"}, artifacts_dir / "gmm_model.joblib")
    (artifacts_dir / "module2_run_summary.json").write_text(
        json.dumps(
            {
                "gmm_selection": {"best_k": 2, "best_covariance_type": "diag"},
                "membership_diagnostics_base": {"prop_high_confidence": 1.0},
                "bootstrap": {"mean_ari": 0.7},
                "cluster_profiles": {"cluster_hard_sizes": {"pi_1": 1, "pi_2": 1}},
            }
        )
    )

    cfg_path = _write_module2_config(tmp_path)
    monkeypatch.setenv("AIREADI_DATA_PATH", str(tmp_path))

    with pytest.raises(ValueError, match="stale relative to the current assembled outcome matrix"):
        promote_solution(cfg_path=cfg_path, slot="primary", view="environment", experiment="stability_v1")


@pytest.mark.xfail(reason="Remaining Module 2 components not yet implemented")
def test_module2_pipeline_placeholder():
    raise NotImplementedError
