# Software Architecture Overview
## Salutogenic Lifestyle Phenotyping in T2DM

---

## Guiding Principles

- **Modules are independent scripts/packages** — no module imports directly from another at runtime. Communication happens through clearly defined data artifacts written to disk
- **Reproducibility by default** — every run is parameterized via config, every output is versioned
- **Separation of concerns** — data transformation, modeling, and reporting never live in the same script
- **Sensitive data paths via environment variables** — `config.yaml` references `${AIREADI_DATA_PATH}` which resolves at runtime from a gitignored `.env` file. The agent sees config structure but no filesystem path to follow. Data never lives inside the repo.
- **Fail loudly** — data contracts between modules are validated explicitly at ingestion; a module should refuse to run on malformed input rather than silently produce bad output

---

## Project Structure

```
# External storage — outside repo root, path set via AIREADI_DATA_PATH env variable
# Agent has no access to this location
${AIREADI_DATA_PATH}/
├── raw/                             # Untouched AI-READI source data (read-only)
├── processed/
│   ├── intermediates/               # One parquet per modality — overwritten on rerun
│   │   ├── wearable_features.parquet
│   │   ├── cgm_features.parquet
│   │   ├── environment_features.parquet
│   │   └── clinical_features.parquet
│   ├── qc_reports/                  # One QC report per modality
│   │   ├── wearable_qc.json
│   │   ├── cgm_qc.json
│   │   ├── environment_qc.json
│   │   └── clinical_qc.json
│   ├── clustering_matrix.parquet    # Wearable + environment only → Module 2
│   └── outcome_matrix.parquet       # CGM + clinical vars → Module 3 directly
└── artifacts/                       # Module 2+ outputs

# Repo root — agent scope limited to here
project_root/
│
├── config/
│   └── config.yaml                  # All parameters — data paths reference ${AIREADI_DATA_PATH}
│
├── module1_processing/
│   ├── wearable_features.py         # Extract wearable features → intermediate
│   ├── cgm_features.py              # Extract CGM features → intermediate
│   ├── environment_features.py      # Extract environment features → intermediate
│   ├── clinical_features.py         # Extract clinical vars (HbA1c, diabetes stage, site) → intermediate
│   ├── assemble.py                  # Inner join intermediates → clustering_matrix + outcome_matrix
│   ├── explore.py                   # PCA fitness checks on clustering_matrix — runs after assemble
│   └── pipeline.py                  # Orchestrates module 1 end-to-end
│
├── module2_clustering/
│   ├── dimensionality_reduction.py
│   ├── gmm_clustering.py
│   ├── bootstrap.py                 # Bootstrap loop + early stopping logic
│   ├── shap_importance.py           # SHAP computation inside bootstrap
│   ├── cluster_profiling.py         # Back-projection, radar plots, profile tables
│   └── pipeline.py
│
├── module3_bayesian/
│   ├── threshold_model.py           # PRIMARY — Bayesian threshold model: P(CV<36% | HbA1c stratum, cluster)
│   ├── residualization.py           # COMPARISON — Stage 1 regression for two-stage model
│   ├── two_stage_model.py           # COMPARISON — two-stage Bayesian model
│   ├── joint_model.py               # COMPARISON — unified single posterior
│   ├── salutogenic_analysis.py      # Exceedance curves, stochastic dominance
│   └── pipeline.py
│
├── module4_reporting/
│   ├── figures.py                   # All figure generation
│   ├── tables.py                    # Summary and demographic tables
│   └── pipeline.py
│
├── tests/
│   ├── test_module1.py
│   ├── test_module2.py
│   └── test_module3.py
│
├── runs/                            # Experiment tracking — one folder per run
│   └── run_YYYYMMDD_HHMMSS/
│       ├── config_snapshot.yaml     # Exact config used
│       ├── logs/
│       └── outputs/
│
├── environment.yml                  # Conda lockfile
├── .env                             # Gitignored — AIREADI_DATA_PATH defined here
├── .env.example                     # Committed — template showing required variables, no values
└── README.md
```

---

## Module 1 Internal Flow & Data Contract

### Per-Modality Scripts
Each modality script is responsible for exactly two things:
1. **Feature extraction** → write `${AIREADI_DATA_PATH}/processed/intermediates/<modality>_features.parquet`
2. **QC reporting** → write `${AIREADI_DATA_PATH}/processed/qc_reports/<modality>_qc.json`

Scripts are independent — rerunning `cgm_features.py` overwrites only `cgm_features.parquet` and `cgm_qc.json`. No other intermediates are touched.

Each intermediate has the schema:
```
<modality>_features.parquet
    - Index: participant_id
    - Columns: engineered features for that modality (raw scale, not normalized)
    - Only participants who passed modality-level QC thresholds are included
      (participants failing QC are excluded here, not at join time)
```

Each QC report captures:
```json
{
  "modality": "cgm",
  "n_input": int,
  "n_passed": int,
  "n_excluded": int,
  "exclusion_reasons": {
    "reason_description": count
  },
  "thresholds_applied": { ... }
}
```

QC thresholds are fully defined in `config.yaml`.

- **Wearable** (`wearable_features.py`) — participant-level summaries plus modality-level valid-hour coverage. These feed `clustering_matrix.parquet`. Per-feature summary strategy:
  - `heart_rate` — median, IQR, resting HR (sleep-period median)
  - `oxygen_saturation` — proportion of hours below 95%, IQR
  - `physical_activity` — median, IQR
  - `calories` — median, IQR (zero is valid per QC; non-wear filtered by HR coverage)
  - `respiratory_rate` — median, IQR (note: sleep-period median may be stronger alternative — deferred pending distribution review)
  - `sleep` — total sleep duration median + IQR, combined deep+REM duration median
  - `stress` — median, IQR (note: structured missingness during exercise — active participants have fewer readings)
- **Environment** (`environment_features.py`) — participant-level summaries plus modality-level valid-hour coverage. These feed `clustering_matrix.parquet`. Per-feature summary strategy:
  - `pm1` — median, IQR
  - `pm2_5` — median, IQR (TODO: verify availability in raw sensor data)
  - `pm10` — median, IQR
  - `humidity` — median, IQR (note: expected correlation with temperature)
  - `temperature` — median, IQR (note: expected correlation with humidity)
  - `voc` — median, IQR, proportion of hours above 150 (meaningful elevation above adaptive baseline)
  - `nox` — proportion of hours above 20 (meaningful elevation per sensor documentation); no median retained
  - `light_total` — total intensity median (sum of lch0–lch11), proportion of hours above data-derived threshold (inspect distribution to define)

**Dropped features:**
  - `pm4` — estimated from PM1/PM2.5 by sensor algorithm, not independently measured; redundant with PM1 and PM2.5
  - `lch0–lch11` — 12 individual spectral channels collapsed into `light_total`; individual channels highly correlated and non-interpretable for lifestyle clustering
  - `screen` — not used as a feature
- **CGM** (`cgm_features.py`) — computes finished outcome features directly from valid glucose observations; no median/IQR. Outputs:
  - `glycemic_cv` (PRIMARY) — (SD / mean) × 100 over valid observations
  - `mean_glucose` — falls out of CV calculation at no extra cost, retained for reporting
  - `time_in_range` (SECONDARY) — proportion of readings 70–180 mg/dL; acknowledged 10-day limitation, available for sensitivity analysis
  These feed `outcome_matrix.parquet` only — never enter clustering.
- **Clinical** (`clinical_features.py`) — HbA1c, HbA1c stratum, diabetes stage. Reserved for downstream severity stratification only — not used as clustering inputs. Feeds `outcome_matrix.parquet`.

### Assembly Script (`assemble.py`)
- Reads all modality intermediates
- **Inner join on participant_id** — only participants present in all modalities and passing QC are retained
- Logs final n after join with breakdown of who was dropped at which modality
- Missing handling: wearable/env missing values are allowed initially; `module1.missing_strategy` controls handling before clustering (default `drop` removes any row with NaNs from clustering/outcome matrices)
- Produces two strictly separated output artifacts:
  - **`clustering_matrix.parquet`** — wearable + environmental features only, normalized. This is the only artifact Module 2 consumes. CGM and clinical variables are explicitly excluded. Rows with missing values are dropped when `module1.missing_strategy: drop`.
  - **`outcome_matrix.parquet`** — CGM-derived glycemic features + clinical variables (HbA1c stratum, diabetes stage), not normalized. Consumed directly by Module 3. Never touches Module 2.
- Normalization applied to `clustering_matrix` only — outcome variables remain in natural units for interpretability
- Log transforms: `log1p` applied pre-normalization to right-skewed clustering features (calories, respiratory_rate, sleep_total*, pm1/pm2.5/pm10, light_total); proportion features are left as-is.
- Stage balance audit written to `assemble_balance.json`; warns if any stage loses >20% of participants across assembly

### Exploration Script (`explore.py`)
Runs after `assemble.py` on `clustering_matrix.parquet`. Purely diagnostic — produces no artifacts consumed downstream.

Checks:
1. **KMO + Bartlett's test** — KMO > 0.6 and Bartlett p < 0.05 required; pipeline halts with error if either fails
2. **Per-feature skewness** — flags features with |skewness| > 2 as candidates for log-transform or rank-normalization
3. **Missingness rate** — flags any feature with > 5% missing values after inner join
4. **Near-zero variance** — flags features with variance < 0.01 post-normalization
5. **Correlation matrix** — flags feature pairs with |r| > 0.85 as potentially redundant; informational only, no pipeline halt. Known expected correlations: stress/HR (HRV-derived overlap), humidity/temperature (physically coupled), NOx/VOC (combustion events), PM channels (cumulative nesting)

Output:
```
${AIREADI_DATA_PATH}/processed/exploration_report.json
    {
      "kmo_score": float,
      "bartlett_p": float,
      "pca_fit_passed": bool,
      "skewed_features": [...],       # |skewness| > 2
      "high_missingness_features": [...],  # > 5% missing
      "low_variance_features": [...], # variance < 0.01
      "high_correlation_pairs": [...],  # |r| > 0.85, informational only
      "created": timestamp
    }
```

### Module 1 → Module 2
```
${AIREADI_DATA_PATH}/processed/clustering_matrix.parquet
    - Index: participant_id
    - Columns: wearable + environmental features ONLY, normalized
    - No CGM or clinical variables
    - No nulls
    - Metadata sidecar: clustering_matrix_meta.json
        {
          "n_participants": int,
          "n_features": int,
          "modalities": ["wearable", "environment"],
          "normalization": "standard_scaler",
          "created": timestamp
        }
```

### Module 1 → Module 3 (direct — bypasses Module 2)
```
${AIREADI_DATA_PATH}/processed/outcome_matrix.parquet
    - Index: participant_id
    - Columns:
        CGM:      glycemic_cv, mean_glucose, time_in_range
        Clinical: hba1c, hba1c_stratum, diabetes_stage, site
    - Not normalized — retained in natural units
    - Participant IDs are aligned with clustering_matrix (same inner join cohort)
```

### Module 2 → Module 3
```
${AIREADI_DATA_PATH}/artifacts/membership_matrix.parquet
    - Index: participant_id
    - Columns: pi_1 .. pi_K (soft membership probabilities, sum to 1)

${AIREADI_DATA_PATH}/artifacts/cluster_profiles.parquet
    - Cluster centroids back-projected to original feature space

${AIREADI_DATA_PATH}/artifacts/bootstrap_report.json
    {
      "K_selected": int,
      "stability_mean_ARI": float,
      "stability_CI": [float, float],
      "B_resamples_run": int,
      "early_stopped": bool,
      "cluster_pvalues": [...]
    }

${AIREADI_DATA_PATH}/artifacts/shap_distributions.parquet
    - Mean SHAP ± 95% CI per feature per cluster across bootstrap resamples
```

### Module 3 → Module 4
```
${AIREADI_DATA_PATH}/artifacts/threshold_posterior.nc        # PRIMARY — NetCDF via ArviZ, threshold model posterior
${AIREADI_DATA_PATH}/artifacts/two_stage_posterior.nc        # COMPARISON — two-stage model posterior
${AIREADI_DATA_PATH}/artifacts/joint_posterior.nc            # COMPARISON — joint model posterior
${AIREADI_DATA_PATH}/artifacts/loo_comparison.csv            # LOO-CV results across all three model variants
${AIREADI_DATA_PATH}/artifacts/exceedance_curves.parquet     # P(CV<36% | HbA1c stratum, cluster) per cluster
${AIREADI_DATA_PATH}/artifacts/cluster_contrasts.csv         # Pairwise posterior contrasts + CIs
```

---

## Configuration

All runtime parameters live in a single `config.yaml`. No magic numbers inside scripts.

```yaml
data:
  raw_path: "${AIREADI_DATA_PATH}/raw/"
  processed_path: "${AIREADI_DATA_PATH}/processed/"
  intermediates_path: "${AIREADI_DATA_PATH}/processed/intermediates/"
  qc_reports_path: "${AIREADI_DATA_PATH}/processed/qc_reports/"
  artifacts_path: "${AIREADI_DATA_PATH}/artifacts/"

module1:
  qc_thresholds:
    wearable:
      min_heart_rate_valid_hour_coverage: 0.70   # Participant excluded if < 70% valid HR hours
      stress_min: 0
      stress_max: 100
      oxygen_saturation_min: 50
      oxygen_saturation_max: 100
      heart_rate_min: 25
      heart_rate_max: 250
      respiratory_rate_min: 4
      respiratory_rate_max: 60
      physical_activity_min: 0        # counts/movement quantity
      calories_min: 0
      sleep_duration_min_hours: 0
      sleep_duration_max_hours: 24
    cgm:
      glucose_min_mg_dl: 40
      glucose_max_mg_dl: 400
    environment:
      pm1_min: 0
      pm1_max: 65536
      pm2_5_min: 0                    # TODO: verify PM2.5 availability in raw sensor data
      pm2_5_max: 65536
      # pm4 dropped — estimated from PM1/PM2.5, not independently measured
      pm10_min: 0
      pm10_max: 65536
      humidity_min: 0
      humidity_max: 100
      temperature_min_c: -10
      temperature_max_c: 50
      voc_min: 0
      voc_max: 500
      nox_min: 0
      nox_max: 500
      light_min: 0.0
      light_max: 1.0
  environment_feature_summaries:
    light_total_activity_threshold: null  # TBD — defined by distribution inspection of summed lch0-lch11
    voc_elevation_threshold: 150          # meaningful elevation above adaptive baseline (index units)
    nox_elevation_threshold: 20           # meaningful elevation per Sensirion documentation (index units)
  normalization: "standard_scaler"   # Applied in assemble.py after join

module2:
  k_range: [3, 4, 5]
  covariance_types: ["full", "diagonal", "tied"]
  bootstrap_B: 1000
  bootstrap_early_stop_threshold: 0.001
  pca_variance_threshold: 0.88
  random_seed: 42

module3:
  # Model hierarchy: threshold_model = primary; two_stage and joint = comparison
  primary_model: "threshold"
  severity_covariates: ["hba1c_stratum", "diabetes_stage", "site"]
  hba1c_strata_boundaries:           # Standard clinical cutoffs — unitless %
    well_controlled: 7.0             # < 7.0%
    moderate: 9.0                    # 7.0–9.0%
    # poor control: > 9.0%
  # diabetes_stage: 4-level categorical (0=no diabetes, 1=prediabetes,
  #                 2=oral/non-insulin injectable, 3=insulin-controlled)
  primary_outcome: "glycemic_cv"
  cv_clinical_threshold: 36.0        # CV < 36% = good glycemic stability
  run_comparison_models: true        # Runs two-stage and joint alongside primary
  sampling:
    draws: 2000
    tune: 1000
    chains: 4
    target_accept: 0.9

reporting:
  figure_format: "pdf"
  dpi: 300
```

---

## Environment Variables

All sensitive paths are resolved from environment variables, never hardcoded. A `.env` file at the repo root is loaded at runtime and is always gitignored.

`.env` (gitignored — never committed):
```bash
AIREADI_DATA_PATH=/path/to/your/local/aireadi_storage
```

`.env.example` (committed — documents required variables without values):
```bash
AIREADI_DATA_PATH=        # Absolute path to your local AI-READI data directory
```

`.gitignore` must include:
```
.env
data/
runs/
*.parquet
*.nc
*.csv
config/secrets.yaml
```

Anyone cloning the repo creates their own `.env` pointing to their locally pulled AI-READI data. No data paths, credentials, or participant data ever enter the repo.

---

## Core Dependencies

| Purpose | Library |
|---|---|
| Data manipulation | `pandas`, `numpy` |
| Parquet I/O | `pyarrow` |
| Dimensionality reduction | `scikit-learn` (PCA, preprocessing) |
| UMAP | `umap-learn` |
| Clustering | `scikit-learn` (GaussianMixture) |
| Bootstrap parallelization | `joblib` |
| SHAP | `shap` |
| Gradient boosted classifier | `lightgbm` or `xgboost` |
| Bayesian modeling | `pymc` |
| Posterior analysis & diagnostics | `arviz` |
| Stage 1 regression | `statsmodels` |
| Figures | `matplotlib`, `seaborn` |
| Environment variables | `python-dotenv` |
| Config parsing | `pyyaml` |
| Environment | `conda` with `environment.yml` lockfile |

---

## Experiment Tracking

Every pipeline run creates a timestamped folder under `runs/`. This captures:
- Exact config snapshot used
- Git commit hash at time of run
- Module runtimes
- Key result summaries (K selected, mean ARI, LOO-CV winner)

This is lightweight — no external tooling like MLflow required unless the grid search grows. A simple Python logging setup writing to `runs/run_YYYYMMDD/logs/` is sufficient.

---

## Parallelization Strategy

| Task | Strategy |
|---|---|
| Bootstrap resamples (Module 2) | `joblib.Parallel` across B iterations — embarrassingly parallel |
| GMM grid search (K × covariance type) | `joblib.Parallel` across configurations |
| SHAP computation per resample | Runs inside bootstrap worker — no additional parallelization needed |
| Bayesian sampling (Module 3) | PyMC runs chains in parallel natively via `cores` argument |

Bootstrap is the most expensive step. At n=2280, K≤5, d~12 (post-PCA), B=1000: estimate ~30–60 min on a local machine with 8 cores. If grid search across K and covariance types is added, multiply by ~9 configurations — at that point a cloud spot instance is worthwhile.

---

## Testing Strategy

Each module has a corresponding test file. Tests operate on a **synthetic mini-dataset** (n=100, same schema as real data) generated once and committed to `tests/fixtures/`. This means tests never touch real data and run fast.

- **Module 1 tests** — schema validation per modality intermediate, QC report structure correctness, inner join logic (assert participants failing any modality are excluded), normalization applied only post-join
- **Module 2 tests** — membership probabilities sum to 1, bootstrap loop runs without error, SHAP output shape correctness
- **Module 3 tests** — threshold model posterior samples valid (R-hat < 1.01), exceedance curve is monotone, LOO-CV table contains all three model variants

---

## Interface Between Modules (Runtime)

Each `pipeline.py` is callable as a standalone script:

```bash
python module1_processing/pipeline.py --config config/config.yaml
python module2_clustering/pipeline.py --config config/config.yaml
python module3_bayesian/pipeline.py   --config config/config.yaml
python module4_reporting/pipeline.py  --config config/config.yaml
```

Each pipeline validates its expected input artifacts exist before running. If upstream artifacts are missing or schema-mismatched, the pipeline exits with a descriptive error. No silent failures.

---

## Residualization Covariate Note

The diabetes staging variable in AI-READI has four categories:
- `0` = No diabetes
- `1` = Prediabetes / lifestyle-controlled
- `2` = Diabetes, oral or non-insulin injectable medications
- `3` = Diabetes, insulin-controlled

This enters the stage 1 regression as 3 dummy variables (reference = no diabetes) or as an ordinal variable if proportional odds holds. It captures treatment intensity as a proxy for disease severity progression.
