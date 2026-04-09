"""Shared utilities for Module 2 clustering.

- Loads config and resolves data/artifact paths using AIREADI_DATA_PATH
- Keeps module self-contained (no imports from Module 1)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv


@dataclass
class Paths:
    """Resolved paths for Module 2 inputs/outputs."""

    data_root: Path
    processed_path: Path
    artifacts_path: Path
    clustering_matrix: Path
    clustering_meta: Path
    runs_path: Path


@dataclass
class ViewPaths:
    """Resolved paths for a named clustering view."""

    view_name: str
    processed_dir: Path
    clustering_matrix: Path
    clustering_meta: Path
    artifacts_path: Path


def load_config(cfg_path: Path) -> Dict[str, Any]:
    """Load YAML config with no mutation."""

    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def resolve_paths(cfg: Dict[str, Any]) -> Paths:
    """Resolve ${AIREADI_DATA_PATH}-templated paths defined in config."""

    load_dotenv()
    data_root = os.getenv("AIREADI_DATA_PATH")
    if not data_root:
        raise EnvironmentError("AIREADI_DATA_PATH not set; define it in .env")

    base = Path(data_root).expanduser()
    processed = _substitute_env(cfg["data"]["processed_path"], base)
    artifacts = _substitute_env(cfg["data"]["artifacts_path"], base)

    return Paths(
        data_root=base,
        processed_path=processed,
        artifacts_path=artifacts,
        clustering_matrix=processed / "clustering_matrix.parquet",
        clustering_meta=processed / "clustering_matrix_meta.json",
        runs_path=Path("runs"),
    )


def default_clustering_view(cfg: Dict[str, Any]) -> str:
    return str(cfg.get("module1", {}).get("clustering_views", {}).get("default_view", "wearable_environment"))


def resolve_view_paths(cfg: Dict[str, Any], view_name: str, experiment_name: str | None = None) -> ViewPaths:
    paths = resolve_paths(cfg)
    processed_dir = paths.processed_path / "clustering_views" / view_name
    artifacts_dir = paths.artifacts_path / "module2" / view_name
    if experiment_name:
        artifacts_dir = artifacts_dir / experiment_name

    return ViewPaths(
        view_name=view_name,
        processed_dir=processed_dir,
        clustering_matrix=processed_dir / "clustering_matrix.parquet",
        clustering_meta=processed_dir / "clustering_matrix_meta.json",
        artifacts_path=artifacts_dir,
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _substitute_env(template: str, data_root: Path) -> Path:
    return Path(template.replace("${AIREADI_DATA_PATH}", str(data_root))).expanduser()
