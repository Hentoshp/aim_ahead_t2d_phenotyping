from __future__ import annotations

import argparse
from pathlib import Path

from poster.common import load_poster_context, write_markdown_table


def make_model_selection_table(cfg_path: Path, slot: str = "primary") -> Path:
    ctx = load_poster_context(cfg_path, slot=slot)
    df = ctx.selection_df.copy()
    keep_cols = [
        "view_name",
        "experiment_name",
        "best_k",
        "best_covariance_type",
        "base_prop_high_confidence",
        "bootstrap_mean_ari",
        "bootstrap_ari_std",
        "smallest_hard_cluster_n",
        "smallest_hard_cluster_fraction",
        "selection_status",
    ]
    out_df = df.loc[:, keep_cols].rename(
        columns={
            "view_name": "view",
            "experiment_name": "experiment",
            "best_k": "K",
            "best_covariance_type": "covariance",
            "base_prop_high_confidence": "base_confidence",
            "bootstrap_mean_ari": "bootstrap_ari",
            "bootstrap_ari_std": "ari_std",
            "smallest_hard_cluster_n": "min_cluster_n",
            "smallest_hard_cluster_fraction": "min_cluster_frac",
            "selection_status": "status",
        }
    )

    csv_path = ctx.tables_dir / "model_selection_summary.csv"
    md_path = ctx.tables_dir / "model_selection_summary.md"
    out_df.to_csv(csv_path, index=False)
    write_markdown_table(out_df, md_path)
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build model selection summary table for poster assets")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--slot", default="primary", help="Selected slot name")
    args = parser.parse_args()
    out_path = make_model_selection_table(Path(args.config), slot=args.slot)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
