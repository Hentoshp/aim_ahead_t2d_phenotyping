from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv, det
from scipy.stats import chi2

from .common import load_config, ensure_dirs


def kmo_bartlett(df: pd.DataFrame) -> tuple[float, float, float, int]:
    """Compute KMO overall statistic and Bartlett's test of sphericity.

    Returns: kmo_overall, bartlett_chi2, bartlett_p, df_bartlett
    """
    data = df.dropna()
    if data.empty:
        raise ValueError("No data left after dropping NaNs for KMO/Bartlett")

    corr = np.corrcoef(data, rowvar=False)
    inv_corr = inv(corr)
    p = corr.shape[0]

    # Partial correlations
    partial = -inv_corr / np.outer(np.sqrt(np.diag(inv_corr)), np.sqrt(np.diag(inv_corr)))
    np.fill_diagonal(partial, 0)

    # KMO
    r2 = corr ** 2
    p2 = partial ** 2
    kmo_num = np.sum(r2) - np.sum(np.diag(r2))
    kmo_den = kmo_num + (np.sum(p2) - np.sum(np.diag(p2)))
    kmo_overall = kmo_num / kmo_den if kmo_den != 0 else 0

    # Bartlett's test (sphericity)
    n = data.shape[0]
    chi2_val = -(n - 1 - (2 * p + 5) / 6) * np.log(det(corr))
    df_bartlett = p * (p - 1) / 2
    p_val = 1 - chi2.cdf(chi2_val, df_bartlett)

    return kmo_overall, chi2_val, p_val, int(df_bartlett)


def plot_skewed_histograms(df: pd.DataFrame, skewed_features: list[str], out_dir: Path) -> dict[str, str]:
    """Save histograms for skewed features and return mapping to file paths."""
    if not skewed_features:
        return {}

    out_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}

    for feature in skewed_features:
        if feature not in df.columns:
            continue

        series = df[feature].dropna()
        if series.empty:
            continue

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(series, bins=30, color="#4C78A8", edgecolor="white")
        ax.set_title(f"{feature} (n={len(series)})")
        ax.set_xlabel(feature)
        ax.set_ylabel("Count")
        fig.tight_layout()

        out_path = out_dir / f"{feature}_hist.png"
        fig.savefig(out_path)
        plt.close(fig)
        saved[feature] = str(out_path)

    return saved


def run(cfg_path: Path) -> None:
    cfg, base = load_config(cfg_path)
    processed_path = Path(cfg["data"]["processed_path"].replace("${AIREADI_DATA_PATH}", str(base)))
    clustering_path = processed_path / "clustering_matrix.parquet"
    raw_path = processed_path / "clustering_matrix_raw.parquet"
    if not clustering_path.exists():
        raise FileNotFoundError(f"Missing clustering matrix: {clustering_path}")

    df = pd.read_parquet(clustering_path)
    raw_df = pd.read_parquet(raw_path) if raw_path.exists() else df.copy()
    if df.empty:
        raise ValueError("Clustering matrix is empty")

    # Missingness
    missing_frac = df.isna().mean()
    high_missing = missing_frac[missing_frac > 0.05].index.tolist()

    # Drop rows with NaN for KMO/Bartlett/skew/variance calculations
    clean_df = df.dropna()
    raw_clean_df = raw_df.loc[clean_df.index]
    raw_clean_df = raw_clean_df.dropna()
    if clean_df.empty:
        raise ValueError("No rows without NaNs for diagnostics; consider imputation")

    # Skewness
    skew = raw_clean_df.skew()
    skewed = skew[skew.abs() > 2].index.tolist()

    # Variance
    var = clean_df.var()
    low_var = var[var < 0.01].index.tolist()

    # KMO and Bartlett
    kmo_overall, chi2_val, p_val, df_bart = kmo_bartlett(clean_df)
    pca_fit_passed = (kmo_overall > 0.6) and (p_val < 0.05)

    hist_dir = processed_path / "exploration_histograms"
    histogram_paths = plot_skewed_histograms(raw_clean_df, skewed, hist_dir)

    report = {
        "kmo_score": float(kmo_overall),
        "bartlett_chi2": float(chi2_val),
        "bartlett_df": df_bart,
        "bartlett_p": float(p_val),
        "pca_fit_passed": bool(pca_fit_passed),
        "skew_basis": "raw" if raw_path.exists() else "scaled",
        "skewed_features": skewed,
        "skewness": {k: float(v) for k, v in skew.items()},
        "high_missingness_features": high_missing,
        "low_variance_features": low_var,
        "histogram_paths": histogram_paths,
        "created": pd.Timestamp.utcnow().isoformat(),
    }

    out_path = processed_path / "exploration_report.json"
    out_path.write_text(json.dumps(report, indent=2))

    # Console warnings
    if high_missing:
        print(f"[WARN] High missingness (>5%): {high_missing}")
    if skewed:
        print("[WARN] Skewed features (|skew|>2):")
        for feature in skewed:
            skew_val = float(skew[feature]) if feature in skew else float("nan")
            print(f"  - {feature}: skew={skew_val:.2f}")
    if low_var:
        print(f"[WARN] Low-variance (<0.01): {low_var}")

    if not pca_fit_passed:
        print(f"[ERROR] PCA fitness failed: KMO={kmo_overall:.3f}, Bartlett p={p_val:.3g}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Clustering matrix exploration diagnostics")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()
    run(Path(args.config))


if __name__ == "__main__":
    main()
