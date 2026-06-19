"""Poster-style CV bin count plot by top cluster.

Reads Module 3's analysis dataset and writes:
- a faceted count plot of glycemic CV bins by top cluster
- a CSV table with the underlying bin counts
"""
from __future__ import annotations

import argparse
import os
import tempfile
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from module3_bayesian.pipeline import (
    STABLE_LABEL,
    TOP_CLUSTER_COL,
    load_config,
    resolve_source,
)


DEFAULT_BIN_WIDTH = 5.0
DEFAULT_FILENAME_STEM = "cv_bin_counts_by_top_cluster"
COLOR_STABLE = "#4C9F70"
COLOR_UNSTABLE = "#C4684C"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot glycemic CV bin counts by top cluster")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--slot", default=None, help="Selected slot name, e.g. primary or sensitivity")
    parser.add_argument("--view", default=None, help="Source view name when not using --slot")
    parser.add_argument("--experiment", default=None, help="Source experiment name when not using --slot")
    parser.add_argument("--bin-width", type=float, default=DEFAULT_BIN_WIDTH, help="CV bin width in percentage points")
    parser.add_argument("--out-path", default=None, help="Optional explicit output path for the figure")
    parser.add_argument(
        "--figure-format",
        default=None,
        help="Optional explicit figure format (defaults to reporting.figure_format or pdf)",
    )
    return parser.parse_args()


def load_analysis_dataset(
    cfg_path: Path,
    slot: str | None = None,
    view: str | None = None,
    experiment: str | None = None,
) -> tuple[dict[str, Any], Path, pd.DataFrame]:
    cfg = load_config(cfg_path)
    source = resolve_source(cfg, slot=slot, view=view, experiment=experiment)
    analysis_path = source.output_dir / "analysis_dataset.parquet"
    if not analysis_path.exists():
        raise FileNotFoundError(
            f"Analysis dataset not found at {analysis_path}. Run module3_bayesian.pipeline first."
        )
    df = pd.read_parquet(analysis_path)
    return cfg, source.output_dir, df


def build_cv_bin_summary(
    analysis_df: pd.DataFrame,
    threshold: float,
    bin_width: float = DEFAULT_BIN_WIDTH,
    cv_col: str = "glycemic_cv",
    cluster_col: str = TOP_CLUSTER_COL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if bin_width <= 0:
        raise ValueError("bin_width must be positive.")
    for required in (cv_col, cluster_col):
        if required not in analysis_df.columns:
            raise ValueError(f"Analysis dataset missing required column: {required}")

    cv = pd.to_numeric(analysis_df[cv_col], errors="raise")
    if cv.isna().any():
        raise ValueError(f"{cv_col} contains missing values.")

    plot_df = analysis_df.copy()
    plot_df[cv_col] = cv.astype(float)
    plot_df[cluster_col] = plot_df[cluster_col].astype(str)
    if STABLE_LABEL not in plot_df.columns:
        plot_df[STABLE_LABEL] = (plot_df[cv_col] < threshold).astype(int)

    edges = anchored_bin_edges(plot_df[cv_col], threshold=threshold, bin_width=bin_width)
    bins = pd.IntervalIndex.from_breaks(edges, closed="left")
    plot_df["cv_bin"] = pd.cut(plot_df[cv_col], bins=bins)

    clusters = sorted(plot_df[cluster_col].unique().tolist(), key=_cluster_sort_key)
    all_pairs = pd.MultiIndex.from_product([clusters, bins], names=[cluster_col, "cv_bin"])

    counts = (
        plot_df.groupby([cluster_col, "cv_bin"], observed=False)
        .size()
        .reindex(all_pairs, fill_value=0)
        .rename("count")
        .reset_index()
    )
    cluster_totals = plot_df[cluster_col].value_counts().reindex(clusters)
    counts["cluster_n"] = counts[cluster_col].map(cluster_totals).astype(int)
    counts["pct_within_cluster"] = np.where(
        counts["cluster_n"] > 0,
        counts["count"] / counts["cluster_n"],
        0.0,
    )
    counts["bin_left"] = counts["cv_bin"].apply(lambda x: float(x.left)).astype(float)
    counts["bin_right"] = counts["cv_bin"].apply(lambda x: float(x.right)).astype(float)
    counts["bin_mid"] = (counts["bin_left"] + counts["bin_right"]) / 2.0
    counts["bin_label"] = counts.apply(
        lambda row: format_bin_label(row["bin_left"], row["bin_right"]),
        axis=1,
    )
    counts["region"] = np.where(counts["bin_right"] <= threshold, "stable_region", "above_threshold")

    cluster_stats = (
        plot_df.groupby(cluster_col, observed=False)
        .agg(
            n_participants=(cluster_col, "size"),
            median_cv=(cv_col, "median"),
            stable_pct=(STABLE_LABEL, "mean"),
            mean_cv=(cv_col, "mean"),
        )
        .reset_index()
    )
    cluster_stats = cluster_stats.sort_values(cluster_col, key=lambda s: s.map(_cluster_sort_key)).reset_index(drop=True)
    return counts, cluster_stats


def anchored_bin_edges(cv: pd.Series, threshold: float, bin_width: float) -> np.ndarray:
    min_val = float(cv.min())
    max_val = float(cv.max())
    if not np.isfinite(min_val) or not np.isfinite(max_val):
        raise ValueError("CV values must be finite.")

    lower_steps = max(1, int(ceil((threshold - min_val) / bin_width)))
    upper_steps = max(1, int(ceil((max_val - threshold) / bin_width)))
    start = threshold - lower_steps * bin_width
    end = threshold + upper_steps * bin_width
    edges = np.arange(start, end + (bin_width * 0.5), bin_width)

    if edges[0] > min_val:
        edges = np.insert(edges, 0, edges[0] - bin_width)
    if edges[-1] <= max_val:
        edges = np.append(edges, edges[-1] + bin_width)
    return edges


def format_bin_label(left: float, right: float) -> str:
    if float(left).is_integer() and float(right).is_integer():
        return f"{int(left)}-{int(right)}"
    return f"{left:.1f}-{right:.1f}"


def plot_cv_bin_summary(
    counts_df: pd.DataFrame,
    cluster_stats_df: pd.DataFrame,
    threshold: float,
    bin_width: float,
    out_path: Path,
) -> Path:
    clusters = cluster_stats_df[TOP_CLUSTER_COL].astype(str).tolist()
    n_clusters = len(clusters)
    ncols = 2 if n_clusters > 1 else 1
    nrows = int(ceil(n_clusters / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(7.5 * ncols, 4.2 * nrows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes_arr = np.atleast_1d(axes).ravel()

    max_count = int(counts_df["count"].max()) if not counts_df.empty else 1
    y_max = max(1, int(ceil(max_count * 1.15)))

    legend_handles = [
        Patch(facecolor=COLOR_STABLE, edgecolor="white", label=f"CV < {threshold:.0f}"),
        Patch(facecolor=COLOR_UNSTABLE, edgecolor="white", label=f"CV ≥ {threshold:.0f}"),
    ]

    for ax, cluster in zip(axes_arr, clusters):
        cluster_counts = counts_df[counts_df[TOP_CLUSTER_COL] == cluster].copy()
        stats_row = cluster_stats_df[cluster_stats_df[TOP_CLUSTER_COL] == cluster].iloc[0]
        colors = [COLOR_STABLE if region == "stable_region" else COLOR_UNSTABLE for region in cluster_counts["region"]]

        ax.bar(
            cluster_counts["bin_mid"],
            cluster_counts["count"],
            width=bin_width * 0.92,
            color=colors,
            edgecolor="white",
            linewidth=0.8,
            align="center",
        )
        ax.axvline(threshold, color="#222222", linestyle="--", linewidth=1.2, alpha=0.9)
        ax.set_title(
            f"{cluster}  |  n={int(stats_row['n_participants'])}\n"
            f"median CV={stats_row['median_cv']:.1f}, stable={stats_row['stable_pct'] * 100:.1f}%",
            fontsize=11,
        )
        ax.set_ylim(0, y_max)
        ax.grid(axis="y", alpha=0.25, linewidth=0.8)

    for ax in axes_arr[n_clusters:]:
        ax.set_visible(False)

    fig.suptitle(
        "Participant Counts Across Glycemic CV Bins by Top Cluster",
        fontsize=16,
        fontweight="bold",
    )
    fig.supxlabel("Glycemic CV (%)")
    fig.supylabel("Participant count")
    fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.98))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def resolve_plot_path(
    output_dir: Path,
    cfg: dict[str, Any],
    explicit_out_path: str | None = None,
    explicit_format: str | None = None,
) -> Path:
    if explicit_out_path:
        return Path(explicit_out_path)
    figure_format = explicit_format or str(cfg.get("reporting", {}).get("figure_format", "pdf"))
    figure_format = figure_format.lstrip(".")
    return output_dir / f"{DEFAULT_FILENAME_STEM}.{figure_format}"


def _cluster_sort_key(cluster_name: str) -> tuple[int, str]:
    suffix = cluster_name.split("_", 1)[1] if "_" in cluster_name else cluster_name
    try:
        return (0, f"{int(suffix):06d}")
    except ValueError:
        return (1, suffix)


def run_plot(
    cfg_path: Path,
    slot: str | None = None,
    view: str | None = None,
    experiment: str | None = None,
    bin_width: float = DEFAULT_BIN_WIDTH,
    out_path: str | None = None,
    figure_format: str | None = None,
) -> tuple[Path, Path]:
    cfg, output_dir, analysis_df = load_analysis_dataset(cfg_path, slot=slot, view=view, experiment=experiment)
    threshold = float(cfg.get("module3", {}).get("cv_clinical_threshold", 36.0))
    counts_df, cluster_stats_df = build_cv_bin_summary(
        analysis_df=analysis_df,
        threshold=threshold,
        bin_width=bin_width,
    )

    plot_path = resolve_plot_path(output_dir, cfg, explicit_out_path=out_path, explicit_format=figure_format)
    csv_path = plot_path.with_suffix(".csv")
    counts_df.to_csv(csv_path, index=False)
    plot_cv_bin_summary(
        counts_df=counts_df,
        cluster_stats_df=cluster_stats_df,
        threshold=threshold,
        bin_width=bin_width,
        out_path=plot_path,
    )
    return plot_path, csv_path


def main() -> None:
    args = parse_args()
    plot_path, csv_path = run_plot(
        cfg_path=Path(args.config),
        slot=args.slot,
        view=args.view,
        experiment=args.experiment,
        bin_width=float(args.bin_width),
        out_path=args.out_path,
        figure_format=args.figure_format,
    )
    print(f"Poster plot written to {plot_path}")
    print(f"Bin count table written to {csv_path}")


if __name__ == "__main__":
    main()
