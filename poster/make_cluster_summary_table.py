from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from poster.common import cluster_display_name, display_feature_name, load_poster_context, write_markdown_table


def _top_centroid_features(profiles_df: pd.DataFrame, top_n: int = 3) -> dict[int, str]:
    out: dict[int, str] = {}
    for cluster, dfc in profiles_df.groupby("cluster"):
        ranked = dfc.reindex(dfc["centroid_value"].abs().sort_values(ascending=False).index).head(top_n)
        parts = [f"{display_feature_name(row.feature)} ({row.centroid_value:+.2f})" for row in ranked.itertuples()]
        out[int(cluster)] = "; ".join(parts)
    return out


def _top_shap_features(shap_df: pd.DataFrame, top_n: int = 3) -> dict[int, str]:
    out: dict[int, str] = {}
    for cluster, dfc in shap_df.groupby("cluster"):
        ranked = dfc.sort_values("abs_shap_mean", ascending=False).head(top_n)
        parts = [f"{display_feature_name(row.feature)} ({row.abs_shap_mean:.3f})" for row in ranked.itertuples()]
        out[int(cluster)] = "; ".join(parts)
    return out


def make_cluster_summary_table(cfg_path: Path, slot: str = "primary") -> Path:
    ctx = load_poster_context(cfg_path, slot=slot)
    counts = ctx.assignments_df["cluster"].value_counts().sort_index()
    total_n = int(counts.sum())

    centroid_summary = _top_centroid_features(ctx.profiles_df, top_n=3)
    shap_summary = _top_shap_features(ctx.shap_summary_df, top_n=3)

    rows = []
    for cluster, n in counts.items():
        rows.append(
            {
                "cluster": cluster_display_name(cluster),
                "n": int(n),
                "fraction": f"{(n / total_n):.1%}",
                "top_profile_features": centroid_summary.get(int(cluster), ""),
                "top_shap_features": shap_summary.get(int(cluster), ""),
                "poster_label": "",
            }
        )
    out_df = pd.DataFrame(rows)

    csv_path = ctx.tables_dir / "cluster_summary_table.csv"
    md_path = ctx.tables_dir / "cluster_summary_table.md"
    out_df.to_csv(csv_path, index=False)
    write_markdown_table(out_df, md_path)
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cluster summary table for poster assets")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--slot", default="primary", help="Selected slot name")
    args = parser.parse_args()
    out_path = make_cluster_summary_table(Path(args.config), slot=args.slot)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
