from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from poster.common import build_profile_matrix, cluster_display_name, display_feature_name, load_poster_context


def plot_cluster_profiles(cfg_path: Path, slot: str = "primary", top_n: int = 12, transpose: bool = True) -> Path:
    ctx = load_poster_context(cfg_path, slot=slot)
    matrix = build_profile_matrix(ctx.profiles_df, top_n=top_n)
    matrix.to_csv(ctx.tables_dir / "cluster_profile_matrix.csv")

    plot_matrix = matrix.T if transpose else matrix

    if transpose:
        fig, ax = plt.subplots(figsize=(5.8, max(6.5, 0.5 * plot_matrix.shape[0])))
    else:
        fig, ax = plt.subplots(figsize=(max(9, 0.8 * plot_matrix.shape[1]), 4.8))

    vmax = float(np.abs(plot_matrix.values).max())
    im = ax.imshow(plot_matrix.values, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(plot_matrix.shape[1]))
    if transpose:
        ax.set_xticklabels([cluster_display_name(col) for col in plot_matrix.columns])
        ax.set_yticks(range(plot_matrix.shape[0]))
        ax.set_yticklabels([display_feature_name(idx) for idx in plot_matrix.index])
    else:
        ax.set_xticklabels([display_feature_name(col) for col in plot_matrix.columns], rotation=35, ha="right")
        ax.set_yticks(range(plot_matrix.shape[0]))
        ax.set_yticklabels([cluster_display_name(idx) for idx in plot_matrix.index])
    ax.set_title("Cluster profiles in standardized feature space")

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Centroid value (z-score)")

    fig.tight_layout()
    out_path = ctx.plots_dir / "cluster_profile_heatmap.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot cluster profile heatmap for poster assets")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--slot", default="primary", help="Selected slot name")
    parser.add_argument("--top-n", type=int, default=12, help="Number of features to include")
    parser.add_argument("--no-transpose", action="store_true", help="Use the original horizontal orientation")
    args = parser.parse_args()
    out_path = plot_cluster_profiles(
        Path(args.config),
        slot=args.slot,
        top_n=args.top_n,
        transpose=not args.no_transpose,
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
