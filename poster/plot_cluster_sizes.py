from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from poster.common import cluster_display_name, load_poster_context


def plot_cluster_sizes(cfg_path: Path, slot: str = "primary") -> Path:
    ctx = load_poster_context(cfg_path, slot=slot)
    assignments = ctx.assignments_df.copy()
    assignments["high_confidence"] = assignments["max_membership"] > 0.7

    counts = assignments["cluster"].value_counts().sort_index()
    high_conf = assignments.groupby("cluster")["high_confidence"].agg(["sum", "mean"]).reset_index()

    counts_df = counts.rename_axis("cluster").reset_index(name="n")
    counts_df["cluster_label"] = counts_df["cluster"].apply(cluster_display_name)
    counts_df["fraction"] = counts_df["n"] / counts_df["n"].sum()
    counts_df = counts_df.merge(high_conf, on="cluster", how="left").rename(
        columns={"sum": "high_confidence_n", "mean": "high_confidence_prop"}
    )
    counts_df["low_confidence_n"] = counts_df["n"] - counts_df["high_confidence_n"]
    counts_df.to_csv(ctx.tables_dir / "cluster_sizes.csv", index=False)

    colors = ["#557C93", "#7AA874", "#C97C5D", "#9C6ADE"]
    fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(7.4, 7.0), height_ratios=[1.15, 1])
    size_ax, conf_ax = axes

    size_bars = size_ax.bar(counts_df["cluster_label"], counts_df["n"], color=colors)
    size_ax.set_title("Base-fit cluster size")
    size_ax.set_ylabel("Participants")
    size_ax.set_ylim(0, counts_df["n"].max() * 1.26)

    for bar, n, frac in zip(size_bars, counts_df["n"], counts_df["fraction"]):
        size_ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + counts_df["n"].max() * 0.03,
            f"{int(n)} ({frac:.1%})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    conf_bars = conf_ax.bar(counts_df["cluster_label"], counts_df["high_confidence_prop"], color=colors)
    conf_ax.set_title("High-confidence assignments")
    conf_ax.set_ylabel("Proportion > 0.7")
    conf_ax.set_ylim(0, 1.10)
    conf_ax.set_xlabel("Cluster")

    for bar, prop, n_high in zip(conf_bars, counts_df["high_confidence_prop"], counts_df["high_confidence_n"]):
        conf_ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.03,
            f"{prop:.0%}\n(n={int(n_high)})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.suptitle("Base-fit cluster size and assignment confidence", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = ctx.plots_dir / "cluster_sizes.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot cluster sizes for poster assets")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--slot", default="primary", help="Selected slot name")
    args = parser.parse_args()
    out_path = plot_cluster_sizes(Path(args.config), slot=args.slot)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
