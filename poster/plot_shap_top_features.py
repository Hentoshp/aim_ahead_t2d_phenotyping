from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from poster.common import cluster_display_name, display_feature_name, load_poster_context


def plot_shap_top_features(cfg_path: Path, slot: str = "primary", top_n: int = 8) -> Path:
    ctx = load_poster_context(cfg_path, slot=slot)
    shap_df = ctx.shap_summary_df.copy()
    top_df = (
        shap_df.sort_values(["cluster", "abs_shap_mean"], ascending=[True, False])
        .groupby("cluster", group_keys=False)
        .head(top_n)
        .copy()
    )
    top_df["feature_label"] = top_df["feature"].map(display_feature_name)
    top_df.to_csv(ctx.tables_dir / "shap_top_features.csv", index=False)

    clusters = sorted(top_df["cluster"].unique())
    ncols = 2
    nrows = (len(clusters) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 3.6 * nrows))
    axes = axes.flatten()

    for ax, cluster in zip(axes, clusters):
        dfc = top_df[top_df["cluster"] == cluster].sort_values("abs_shap_mean", ascending=True)
        ax.barh(dfc["feature_label"], dfc["abs_shap_mean"], color="#3E7CB1")
        ax.set_title(cluster_display_name(cluster))
        ax.set_xlabel("Mean |SHAP|")
    for ax in axes[len(clusters):]:
        ax.axis("off")

    fig.suptitle("Top SHAP drivers by cluster", y=0.995, fontsize=14)
    fig.tight_layout()
    out_path = ctx.plots_dir / "shap_top_features.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot top SHAP features for poster assets")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--slot", default="primary", help="Selected slot name")
    parser.add_argument("--top-n", type=int, default=8, help="Top features per cluster")
    args = parser.parse_args()
    out_path = plot_shap_top_features(Path(args.config), slot=args.slot, top_n=args.top_n)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
