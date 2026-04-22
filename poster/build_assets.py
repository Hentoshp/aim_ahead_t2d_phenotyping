from __future__ import annotations

import argparse
from pathlib import Path

from poster.make_cluster_summary_table import make_cluster_summary_table
from poster.make_model_selection_table import make_model_selection_table
from poster.plot_cluster_profile_lines import plot_cluster_profile_lines
from poster.plot_cluster_profile_radar import plot_cluster_profile_radar
from poster.plot_cluster_profiles import plot_cluster_profiles
from poster.plot_cluster_sizes import plot_cluster_sizes
from poster.plot_shap_top_features import plot_shap_top_features
from poster.plot_umap_projection import plot_umap_projection


def build_assets(cfg_path: Path, slot: str = "primary") -> list[Path]:
    outputs = [
        plot_cluster_profiles(cfg_path, slot=slot),
        plot_cluster_profile_lines(cfg_path, slot=slot),
        plot_cluster_profile_radar(cfg_path, slot=slot),
        plot_cluster_sizes(cfg_path, slot=slot),
        plot_shap_top_features(cfg_path, slot=slot),
        plot_umap_projection(cfg_path, slot=slot),
        make_model_selection_table(cfg_path, slot=slot),
        make_cluster_summary_table(cfg_path, slot=slot),
    ]
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build poster plots and tables from selected clustering outputs")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--slot", default="primary", help="Selected slot name")
    args = parser.parse_args()
    outputs = build_assets(Path(args.config), slot=args.slot)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
