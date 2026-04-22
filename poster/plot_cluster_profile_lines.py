from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from poster.common import build_profile_matrix, cluster_display_name, display_feature_name, load_poster_context


COLORS = ["#557C93", "#7AA874", "#C97C5D", "#9C6ADE"]


def plot_cluster_profile_lines(cfg_path: Path, slot: str = "primary", top_n: int = 10) -> Path:
    ctx = load_poster_context(cfg_path, slot=slot)
    matrix = build_profile_matrix(ctx.profiles_df, top_n=top_n)

    fig, ax = plt.subplots(figsize=(max(10, 0.85 * matrix.shape[1]), 5))
    x = range(matrix.shape[1])
    for color, (cluster, row) in zip(COLORS, matrix.iterrows()):
        ax.plot(x, row.values, marker="o", linewidth=2, color=color, label=cluster_display_name(cluster))

    ax.axhline(0, color="#666666", linewidth=1, linestyle="--")
    ax.set_xticks(list(x))
    ax.set_xticklabels([display_feature_name(col) for col in matrix.columns], rotation=35, ha="right")
    ax.set_ylabel("Centroid value (z-score)")
    ax.set_title("Cluster profile line plot")
    ax.legend(frameon=False, ncol=min(4, len(matrix.index)))

    fig.tight_layout()
    out_path = ctx.plots_dir / "cluster_profile_lines.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot cluster profile line plot for poster assets")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--slot", default="primary", help="Selected slot name")
    parser.add_argument("--top-n", type=int, default=10, help="Number of features to include")
    args = parser.parse_args()
    out_path = plot_cluster_profile_lines(Path(args.config), slot=args.slot, top_n=args.top_n)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
