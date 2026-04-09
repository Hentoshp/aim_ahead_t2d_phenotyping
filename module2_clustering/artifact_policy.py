"""Artifact policy helpers for Module 2."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Module2ArtifactPolicy:
    level: str
    save_plots: bool
    save_pca_summary: bool
    save_pca_transformed: bool
    save_pca_loadings: bool
    save_pruned_matrix: bool
    save_debug_sidecars: bool
    save_json_mirrors: bool
    compute_shap: bool
    save_shap_outputs: bool


def resolve_module2_artifact_policy(cfg: dict[str, Any]) -> Module2ArtifactPolicy:
    raw_cfg = cfg.get("module2", {}).get("artifacts", {})
    level = str(raw_cfg.get("level", "standard"))

    defaults_by_level = {
        "standard": {
            "save_plots": False,
            "save_pca_summary": False,
            "save_pca_transformed": False,
            "save_pca_loadings": False,
            "save_pruned_matrix": False,
            "save_debug_sidecars": False,
            "save_json_mirrors": False,
            "compute_shap": False,
            "save_shap_outputs": False,
        },
        "debug": {
            "save_plots": True,
            "save_pca_summary": True,
            "save_pca_transformed": True,
            "save_pca_loadings": True,
            "save_pruned_matrix": True,
            "save_debug_sidecars": True,
            "save_json_mirrors": True,
            "compute_shap": True,
            "save_shap_outputs": True,
        },
    }
    if level not in defaults_by_level:
        raise ValueError(f"Unknown module2 artifact level: {level}")

    resolved = defaults_by_level[level] | {k: v for k, v in raw_cfg.items() if k != "level"}
    if resolved["compute_shap"] and "save_shap_outputs" not in raw_cfg:
        resolved["save_shap_outputs"] = True

    return Module2ArtifactPolicy(level=level, **resolved)
