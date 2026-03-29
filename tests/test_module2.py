import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from module2_clustering.dimensionality_reduction import PCAResult, run_pca, save_pca_artifacts


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


@pytest.mark.xfail(reason="Remaining Module 2 components not yet implemented")
def test_module2_pipeline_placeholder():
    raise NotImplementedError
