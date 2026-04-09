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
│   ├── clustering_views/
│   │   ├── wearable/
│   │   │   ├── clustering_matrix.parquet
│   │   │   ├── clustering_matrix_raw.parquet
│   │   │   └── clustering_matrix_meta.json
│   │   ├── environment/
│   │   └── wearable_environment/
│   ├── outcome_matrix.parquet       # CGM + clinical vars → Module 3 directly
│   ├── outcome_matrix_meta.json
│   ├── assemble_balance.json
│   └── clustering_matrix*.parquet    # Optional root default-view aliases in debug / compatibility mode
└── artifacts/                       # Module 2+ outputs
    └── module2/
        ├── experiment_comparison.csv
        ├── selection_summary.csv
        ├── wearable/
        ├── environment/
        └── wearable_environment/

# Repo root — agent scope limited to here
project_root/
│
├── data_processing.py              # Legacy standalone processing script; not part of current modular pipeline
├── config/
│   └── config.yaml                  # All parameters — data paths reference ${AIREADI_DATA_PATH}
│
├── module1_processing/
│   ├── common.py                    # Shared loaders, JSON/CSV wrappers, feature pull helpers
│   ├── wearable_features.py         # Extract wearable features → intermediate
│   ├── cgm_features.py              # Extract CGM features → intermediate
│   ├── environment_features.py      # Extract environment features → intermediate
│   ├── clinical_features.py         # Extract clinical vars (HbA1c, diabetes stage, site) → intermediate
│   ├── assemble.py                  # Inner join intermediates → clustering_matrix + outcome_matrix
│   ├── explore.py                   # PCA fitness checks on a clustering view — runs after assemble
│   └── pipeline.py                  # Orchestrates module 1 end-to-end
│
├── module2_clustering/
│   ├── utils.py
│   ├── dimensionality_reduction.py
│   ├── gmm_clustering.py
│   ├── bootstrap.py                 # Bootstrap loop + early stopping logic
│   ├── shap_importance.py           # SHAP computation inside bootstrap
│   ├── cluster_profiling.py         # Back-projection, radar plots, profile tables
│   ├── diagnostics.py               # Standalone PCA/GMM diagnostic sweep
│   ├── experiment_runner.py         # Runs view x config experiments and writes comparison tables
│   └── pipeline.py
│
├── module3_bayesian/
│   └── pipeline.py                  # Stub — validates config path then raises NotImplementedError
│
├── module4_reporting/
│   └── pipeline.py                  # Stub — validates config path then raises NotImplementedError
│
├── tests/
│   ├── test_module1.py
│   ├── test_module2.py
│   └── test_module3.py
│
├── runs/                            # Experiment tracking — one folder per run
│   └── run_YYYYMMDD_HHMMSS/
│       └── config_snapshot.yaml     # Exact config used (Module 1 pipeline)
│
├── environment.yml                  # Conda environment definition
└── .env                             # Gitignored — AIREADI_DATA_PATH defined here locally
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
  - `sleep` — total sleep duration median + IQR, combined deep+REM duration median (no per-stage medians/IQRs)
  - `stress` — median, IQR (note: structured missingness during exercise — active participants have fewer readings)
- **Environment** (`environment_features.py`) — participant-level summaries plus modality-level valid-hour coverage. These feed `clustering_matrix.parquet`. Per-feature summary strategy:
  - `pm10` — median, IQR (pm1/pm2.5 dropped as redundant with pm10)
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
- Missing handling: wearable/env missing values are allowed initially; `module1.missing_strategy` controls handling before clustering (default `drop` removes any row with NaNs from the common clustering cohort and aligned outcome matrix)
- Builds named clustering views on the same common cohort, driven by `module1.clustering_views` in `config.yaml`
- Produces two strictly separated output artifact families:
  - **`processed/clustering_views/<view>/clustering_matrix.parquet`** — normalized clustering input for a named view (`wearable`, `environment`, `wearable_environment`). CGM and clinical variables are explicitly excluded.
  - **`processed/clustering_views/<view>/clustering_matrix_meta.json`** — view-level metadata and artifact policy.
  - **`outcome_matrix.parquet`** — CGM-derived glycemic features + clinical variables (HbA1c stratum, diabetes stage), not normalized. Consumed directly by Module 3. Never touches Module 2.
  - **Optional compatibility aliases** — `processed/clustering_matrix*.{parquet,json}` and `processed/clustering_matrix_common_raw.parquet` are written only when `module1.artifacts.write_default_aliases` / debug outputs are enabled.
- Normalization applied to `clustering_matrix` only — outcome variables remain in natural units for interpretability
- Log transforms: `log1p` applied pre-normalization to right-skewed clustering features (calories, respiratory_rate, sleep_total*, pm10, light_total); proportion features are left as-is.
- Stage balance audit written to `assemble_balance.json`; warns if any stage loses >20% of participants across assembly

### Exploration Script (`explore.py`)
Runs after `assemble.py` on the configured default view by default, or on a named view via `--view`. Purely diagnostic — produces no artifacts consumed downstream.

Checks:
1. **KMO + Bartlett's test** — KMO > 0.6 and Bartlett p < 0.05 required; pipeline halts with error if either fails
2. **Per-feature skewness** — flags features with |skewness| > 2 as candidates for log-transform or rank-normalization
3. **Missingness rate** — flags any feature with > 5% missing values after inner join
4. **Near-zero variance** — flags features with variance < 0.01 post-normalization
5. **Correlation matrix** — flags feature pairs with |r| > 0.85 as potentially redundant; informational only, no pipeline halt. Known expected correlations: stress/HR (HRV-derived overlap), humidity/temperature (physically coupled), NOx/VOC (combustion events), PM channels (cumulative nesting)

Output:
```
${AIREADI_DATA_PATH}/processed/clustering_views/<view>/exploration_report.json
    - Written for the selected view (including the default view when no `--view` is passed)
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
${AIREADI_DATA_PATH}/processed/clustering_views/<view>/clustering_matrix.parquet
    - Index: participant_id
    - Columns: view-specific wearable/environment features ONLY, normalized
    - No CGM or clinical variables
    - No nulls
    - Shared common cohort across all views
    - Metadata sidecar: clustering_matrix_meta.json
        {
          "view_name": str,
          "cohort_policy": "common",
          "n_participants": int,
          "n_features": int,
          "feature_names": [...],
          "modalities": [...],
          "normalization": {...},
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
${AIREADI_DATA_PATH}/artifacts/module2/<view>/<experiment>/membership_matrix.parquet
    - Index: participant_id
    - Columns: pi_1 .. pi_K (soft membership probabilities, sum to 1)

${AIREADI_DATA_PATH}/artifacts/module2/<view>/<experiment>/cluster_profiles.parquet
    - Cluster centroids back-projected to original feature space

${AIREADI_DATA_PATH}/artifacts/module2/<view>/<experiment>/gmm_grid_search.csv  # Full GMM grid search table
${AIREADI_DATA_PATH}/artifacts/module2/<view>/<experiment>/module2_run_summary.json
    - Canonical per-run evidence record (config, artifact policy, pruning, PCA, full GMM grid, base diagnostics, bootstrap, profiles, optional SHAP summary)
${AIREADI_DATA_PATH}/artifacts/module2/<view>/<experiment>/pca_model.joblib
    - PCA model needed to reproduce transforms/back-projection

Debug-only / optional artifacts (when `module2.artifacts.level: debug` or explicitly enabled):
${AIREADI_DATA_PATH}/artifacts/module2/<view>/<experiment>/bootstrap_summary.json
${AIREADI_DATA_PATH}/artifacts/module2/<view>/<experiment>/corr_pruned_features.json
${AIREADI_DATA_PATH}/artifacts/module2/<view>/<experiment>/membership_diagnostics_base.json
${AIREADI_DATA_PATH}/artifacts/module2/<view>/<experiment>/membership_diagnostics.json
${AIREADI_DATA_PATH}/artifacts/module2/<view>/<experiment>/pca_summary.json
${AIREADI_DATA_PATH}/artifacts/module2/<view>/<experiment>/pca_transformed.parquet
${AIREADI_DATA_PATH}/artifacts/module2/<view>/<experiment>/pca_loadings.parquet
${AIREADI_DATA_PATH}/artifacts/module2/<view>/<experiment>/gmm_bic_curve.png
${AIREADI_DATA_PATH}/artifacts/module2/<view>/<experiment>/shap_distributions.parquet
${AIREADI_DATA_PATH}/artifacts/module2/<view>/<experiment>/shap_report.json

${AIREADI_DATA_PATH}/artifacts/module2/experiment_comparison.csv
    - One row per view/experiment candidate, suitable for model selection and downstream reporting tables
${AIREADI_DATA_PATH}/artifacts/module2/selection_summary.csv
    - Selection-rule evaluation across candidates (viability, cluster balance, final status)
Debug-only mirrors:
${AIREADI_DATA_PATH}/artifacts/module2/experiment_comparison.json
${AIREADI_DATA_PATH}/artifacts/module2/selection_summary.json
```

### Module 3 → Module 4
```
Module 3 and Module 4 are currently stubs.
- `module3_bayesian/pipeline.py` exists but raises `NotImplementedError`
- `module4_reporting/pipeline.py` exists but raises `NotImplementedError`
- No Module 3 or Module 4 artifacts are produced yet
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
  artifacts:
    level: "standard"
    write_default_aliases: false
    save_view_raw_matrices: true
    save_common_raw_matrix: false
  clustering_views:
    cohort_policy: "common"
    default_view: "wearable_environment"
    views:
      wearable:
        include_prefixes: ["heart_rate_", "oxygen_sat_", "physical_activity_", "calories_", "respiratory_rate_", "stress_", "sleep_"]
      environment:
        include_prefixes: ["env_"]
      wearable_environment:
        include_prefixes: ["heart_rate_", "oxygen_sat_", "physical_activity_", "calories_", "respiratory_rate_", "stress_", "sleep_", "env_"]
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
  k_range: [3, 4]
  covariance_types: ["diag", "tied"]
  bootstrap_B: 1000
  bootstrap_early_stop_threshold: 0.001
  pca_variance_threshold: 0.80
  pca_mode: "variance"
  pca_n_components: null
  corr_prune: true
  corr_threshold: 0.9
  gmm_reg_covar: 0.001
  random_seed: 42
  artifacts:
    level: "standard"
    compute_shap: false
    save_json_mirrors: false
  exploration:
    views: ["wearable", "environment", "wearable_environment"]
    experiments:
      - name: "stability_v1"
        k_range: [3, 4]
        covariance_types: ["diag", "tied"]
        pca_mode: "variance"
        pca_variance_threshold: 0.80
        gmm_reg_covar: 0.001
      - name: "k3_only"
        k_range: [3]
        covariance_types: ["diag", "tied"]
        pca_mode: "variance"
        pca_variance_threshold: 0.80
        gmm_reg_covar: 0.001
  selection:
    min_base_prop_high_confidence: 0.70
    min_bootstrap_mean_ari: 0.50
    preferred_min_cluster_fraction: 0.10
    acceptable_min_cluster_fraction: 0.05
    view_priority: ["wearable_environment", "environment", "wearable"]

module3:
  # Reserved for future implementation; current pipeline is a stub
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

`.gitignore` must include:
```
.env
processed/
artifacts/
runs/
ai-readi-dataset-container/
```

The current `.gitignore` is broader than the abbreviated list above and excludes local data, artifacts, run outputs, and common generated file types.

---

## Core Dependencies

| Purpose | Library |
|---|---|
| Data manipulation | `pandas`, `numpy` |
| Parquet I/O | `pyarrow` |
| Dimensionality reduction | `scikit-learn` (PCA, preprocessing) |
| Clustering | `scikit-learn` (GaussianMixture) |
| Bootstrap parallelization | `joblib` |
| SHAP | `shap` |
| Figures / diagnostics | `matplotlib` |
| Environment variables | `python-dotenv` |
| Config parsing | `pyyaml` |
| Test runner | `pytest` |

`environment.yml` currently also pins `umap-learn`, `lightgbm`, `xgboost`, `pymc`, `arviz`, `statsmodels`, and `seaborn` for planned downstream work, but those libraries are not used by the currently implemented pipelines.

---

## Experiment Tracking

Current behavior:
- `module1_processing/pipeline.py` creates `runs/run_YYYYMMDD_HHMMSS/config_snapshot.yaml`
- Module 2 experiment outputs are written under `${AIREADI_DATA_PATH}/artifacts/module2/...`
- Artifact verbosity is config-driven:
  - `standard` keeps research evidence and reproducibility artifacts
  - `debug` additionally keeps intermediate matrices, sidecar diagnostics, plot outputs, and JSON mirrors
- Git commit hashes, structured logs, and runtime manifests are not currently captured automatically

---

## Parallelization Strategy

| Task | Strategy |
|---|---|
| Bootstrap resamples (Module 2) | `joblib.Parallel` across B iterations |
| SHAP computation per resample | Runs inside each bootstrap worker |
| GMM grid search | Current implementation is serial within a run |
| Module 3 sampling | Not implemented yet |

---

## Testing Strategy

Current test coverage is limited:
- `tests/test_module1.py` is a placeholder and currently skips
- `tests/test_module2.py` covers PCA behavior and artifact writing
- `tests/test_module3.py` is an expected-failure stub because Module 3 is not implemented
- No committed fixture dataset exists in `tests/fixtures/`

---

## Interface Between Modules (Runtime)

Each `pipeline.py` is callable as a standalone script:

```bash
python -m module1_processing.pipeline --config config/config.yaml
python -m module2_clustering.pipeline --config config/config.yaml
python -m module2_clustering.experiment_runner --config config/config.yaml
python -m module3_bayesian.pipeline   --config config/config.yaml
python -m module4_reporting.pipeline  --config config/config.yaml
```

Implemented pipelines validate expected input artifacts before running. Module 3 and Module 4 currently stop with `NotImplementedError` after config-path validation.

---

## Module 3 Note

The diabetes staging variable configured for future Module 3 work has four categories:
- `0` = No diabetes
- `1` = Prediabetes / lifestyle-controlled
- `2` = Diabetes, oral or non-insulin injectable medications
- `3` = Diabetes, insulin-controlled

Those covariates are present in `config.yaml`, but the actual Bayesian model code has not been implemented yet.
