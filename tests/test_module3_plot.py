from pathlib import Path

import pandas as pd

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from module3_bayesian.plot_cv_bins import build_cv_bin_summary, plot_cv_bin_summary


def _make_analysis_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "glycemic_cv": [24.0, 28.0, 33.0, 37.0, 41.0, 46.0, 29.0, 35.0],
            "top_cluster": ["pi_1", "pi_1", "pi_1", "pi_1", "pi_2", "pi_2", "pi_2", "pi_2"],
            "stable": [1, 1, 1, 0, 0, 0, 1, 1],
        },
        index=[f"p{i:02d}" for i in range(8)],
    )


def test_build_cv_bin_summary_anchors_bins_on_threshold():
    df = _make_analysis_df()
    counts_df, cluster_stats_df = build_cv_bin_summary(df, threshold=36.0, bin_width=5.0)

    assert not counts_df.empty
    assert set(cluster_stats_df["top_cluster"].tolist()) == {"pi_1", "pi_2"}
    assert (counts_df["bin_right"] == 36.0).any()

    threshold_edge_rows = counts_df[counts_df["bin_right"] == 36.0]
    assert (threshold_edge_rows["region"] == "stable_region").all()
    above_threshold_rows = counts_df[counts_df["bin_left"] >= 36.0]
    assert (above_threshold_rows["region"] == "above_threshold").all()


def test_plot_cv_bin_summary_writes_file(tmp_path: Path):
    df = _make_analysis_df()
    counts_df, cluster_stats_df = build_cv_bin_summary(df, threshold=36.0, bin_width=5.0)
    out_path = tmp_path / "cv_bins_test.png"

    plot_cv_bin_summary(
        counts_df=counts_df,
        cluster_stats_df=cluster_stats_df,
        threshold=36.0,
        bin_width=5.0,
        out_path=out_path,
    )

    assert out_path.exists()
