from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from poster.common import cluster_display_name, load_poster_context, load_selected_feature_matrix


COLORS = ["#557C93", "#7AA874", "#C97C5D", "#9C6ADE"]


def plot_umap_projection(cfg_path: Path, slot: str = "primary", n_neighbors: int = 30, min_dist: float = 0.1) -> Path:
    ctx = load_poster_context(cfg_path, slot=slot)
    X = load_selected_feature_matrix(ctx)
    assignments = ctx.assignments_df.loc[X.index].copy()

    pca_model_path = Path(ctx.manifest["source_artifacts_path"]) / "pca_model.joblib"
    pca_model = joblib.load(pca_model_path)
    X_pca = pca_model.transform(X.values)

    try:
        import umap
    except ImportError as exc:
        raise ImportError("umap-learn is required for the UMAP poster plot.") from exc

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=int(ctx.run_summary["config"]["random_seed"]),
    )
    coords = reducer.fit_transform(X_pca)

    coords_df = pd.DataFrame(
        {
            "participant_id": X.index,
            "umap_1": coords[:, 0],
            "umap_2": coords[:, 1],
            "cluster": assignments["cluster"].values,
        }
    )
    coords_df.to_csv(ctx.tables_dir / "umap_coordinates.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for color, cluster in zip(COLORS, sorted(coords_df["cluster"].unique())):
        dfc = coords_df[coords_df["cluster"] == cluster]
        ax.scatter(dfc["umap_1"], dfc["umap_2"], s=14, alpha=0.75, color=color, label=cluster_display_name(cluster))

    ax.set_title("UMAP projection of selected clustering input")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(frameon=False, title="Cluster")

    fig.tight_layout()
    out_path = ctx.plots_dir / "umap_projection.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot UMAP projection for poster assets")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--slot", default="primary", help="Selected slot name")
    parser.add_argument("--n-neighbors", type=int, default=30, help="UMAP n_neighbors")
    parser.add_argument("--min-dist", type=float, default=0.1, help="UMAP min_dist")
    args = parser.parse_args()
    out_path = plot_umap_projection(
        Path(args.config),
        slot=args.slot,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
