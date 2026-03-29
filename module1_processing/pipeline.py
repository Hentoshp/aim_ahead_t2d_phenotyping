from __future__ import annotations

import argparse
import time
from pathlib import Path
import shutil
import datetime

from .wearable_features import build_wearable_features
from .cgm_features import build_cgm_features
from .environment_features import build_environment_features
from .clinical_features import build_clinical_features
from .assemble import assemble
from .common import load_config, ensure_dirs


def snapshot_config(cfg_path: Path, runs_dir: Path) -> None:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_dir / f"run_{ts}"
    ensure_dirs(run_dir)
    dest = run_dir / "config_snapshot.yaml"
    shutil.copy2(cfg_path, dest)


def main():
    parser = argparse.ArgumentParser(description="Module 1 processing pipeline")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--runs_dir", default="runs", help="Local runs directory (relative to repo root)")
    parser.add_argument(
        "--steps",
        default="wearable,cgm,environment,clinical,assemble,explore",
        help="Comma-separated steps to run: wearable,cgm,environment,clinical,assemble,explore",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a step if its expected output parquet already exists",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    runs_dir = Path(args.runs_dir)
    ensure_dirs(runs_dir)

    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    step_funcs = {
        "wearable": (build_wearable_features, "wearable_features.parquet", "intermediates_path"),
        "cgm": (build_cgm_features, "cgm_features.parquet", "intermediates_path"),
        "environment": (build_environment_features, "environment_features.parquet", "intermediates_path"),
        "clinical": (build_clinical_features, "clinical_features.parquet", "intermediates_path"),
        "assemble": (assemble, None, None),
        "explore": (lambda cfg: (__import__('module1_processing.explore', fromlist=['run']).run(cfg)), None, None),
    }

    cfg, base = load_config(cfg_path)  # validates env + config and reuse paths

    for step in steps:
        if step not in step_funcs:
            raise ValueError(f"Unknown step '{step}'. Valid: {list(step_funcs)}")
        func, outfile, path_key = step_funcs[step]
        if args.skip_existing and outfile and path_key:
            out_dir = Path(cfg["data"][path_key].replace("${AIREADI_DATA_PATH}", str(base)))
            if (out_dir / outfile).exists():
                continue
        t0 = time.perf_counter()
        func(cfg_path)
        dt = time.perf_counter() - t0
        print(f"[INFO] Step '{step}' completed in {dt:.2f}s")

    snapshot_config(cfg_path, runs_dir)


if __name__ == "__main__":
    main()
