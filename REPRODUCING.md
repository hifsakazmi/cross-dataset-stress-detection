# Reproducing the Results

End-to-end recipe to go from a fresh clone to the final Phase 5 (within-dataset) and Phase 6 (cross-dataset transfer) results. Every command runs from the repo root.

## 0. Prerequisites

- **Python 3.10 or 3.11.** Tested on Windows (PowerShell) and Linux. No GPU needed — everything runs on CPU.
- **Disk:** ~20 GB free. Raw WESAD alone is 16 GB unzipped (we only need the 80 MB of E4 data, but `scripts/extract_data.py` unzips selectively).

## 1. Setup

```
git clone <repo-url> cross-dataset-stress-detection
cd cross-dataset-stress-detection

python -m venv stress-env
# Linux/macOS:    source stress-env/bin/activate
# Windows (PS):   .\stress-env\Scripts\Activate.ps1

pip install -r requirements.txt
```

`requirements.txt` pulls scipy, neurokit2, cvxopt (needed for cvxEDA), scikit-learn, xgboost, pandas. On Windows, `xgboost` installs the CPU wheel — no GPU build required.

## 2. Get the raw data

See `data/README.md` for download links. Place the source archives under `data/`:

```
data/
├── WESAD.zip                                  # https://archive.ics.uci.edu/dataset/465
├── campanella/                                # Mendeley Data: 10.17632/...
│   └── <29 subject folders, as distributed>
└── nurse/
    ├── Stress_dataset.zip                    # Hosseini 2022 — figshare/Zenodo
    └── SurveyResults.xlsx
```

Then extract everything into the flat layout the loaders expect:

```
python -m scripts.extract_data
```

This walks the nested zips and writes:
- `data_extracted/wesad/S2/`, `data_extracted/wesad/S3/`, ... (15 subject folders, E4 CSVs each)
- `data_extracted/campanella/subject_01/`, ... (29 folders)
- `data_extracted/nurse/{nurse_id}_{session_unix_ts}/` (609 session folders) + `data_extracted/nurse/SurveyResults.xlsx`

**Sanity check at this point:**

```
python -m tests.test_preprocessing
```

Loads one subject per dataset, runs preprocessing, prints stats before and after, and confirms Campanella BVP trims exactly 11 samples and WESAD TEMP clips 382 → 40. Should print ✓ for all three datasets in ~10 seconds.

## 3. Extract labels

WESAD and Nurse labels come from auxiliary files (pickles and surveys, respectively). Campanella labels are computed from protocol timing, no extraction step.

```
python -m scripts.extract_wesad_labels
python -m scripts.extract_nurse_labels
```

Writes `labels/wesad/{subject}.csv` (15 files) and `labels/nurse/{session_id}.csv` (34 files — most of the 609 sessions have no aligned surveys and are skipped).

## 4. Feature extraction — six runs

Phase 4 produced one `features.csv`; Phase 6 added five normalization variants. For full reproduction you want all six. **Runtime: ~7 min × 6 ≈ 70 min.**

```
# Phase 4 / Phase 5 baseline — no signal normalization
python -m scripts.extract_features --normalize-mode none \
  --output data_extracted/features_unnormalized.csv

# Phase 6 — per-subject signal z-score (one file covers all sources)
python -m scripts.extract_features --normalize-mode per_subject \
  --output data_extracted/features_per_subject.csv

# Phase 6 — global signal z-score, one transform fit per source pool
python -m scripts.extract_features --normalize-mode global --source-datasets wesad \
  --output data_extracted/features_global_src_wesad.csv
python -m scripts.extract_features --normalize-mode global --source-datasets campanella \
  --output data_extracted/features_global_src_campanella.csv
python -m scripts.extract_features --normalize-mode global --source-datasets nurse \
  --output data_extracted/features_global_src_nurse.csv
python -m scripts.extract_features --normalize-mode global --source-datasets wesad campanella \
  --output data_extracted/features_global_src_wesad_campanella.csv
```

All six should be 4483 rows × 46 columns. Verify:

```
python -c "import pandas as pd; [print(f, pd.read_csv(f).shape) for f in ['data_extracted/features_unnormalized.csv','data_extracted/features_per_subject.csv','data_extracted/features_global_src_wesad.csv','data_extracted/features_global_src_campanella.csv','data_extracted/features_global_src_nurse.csv','data_extracted/features_global_src_wesad_campanella.csv']]"
```

If any row count differs, normalization dropped or duplicated windows — stop and investigate before continuing.

**Sanity check for the new feature variants:**

```
python -m tests.test_normalization_pipeline   # ~30 sec
python -m tests.test_features                  # ~10 sec
```

## 5. Phase 5 — within-dataset LOSO baselines

Phase 5 uses the unnormalized features.

```
python -m tests.test_models                    # ~5 sec, includes regression tests

python -m scripts.run_within_dataset \
  --features-csv data_extracted/features_unnormalized.csv
```

Writes:
- `results/within_dataset_loso.csv` — per-fold + POOLED rows
- `results/within_dataset_summary.csv` — aggregated

Expected pooled macro-F1 (from `within_dataset_summary.csv`):

| Dataset | RF | SVM | XGB |
|---|---|---|---|
| Campanella | **0.684** | 0.663 | 0.683 |
| WESAD | 0.742 | **0.758** | 0.744 |
| Nurse | 0.398 | **0.439** | 0.385 |

## 6. Phase 6 — cross-dataset transfer (84 experiments)

```
python -m tests.test_transfer                  # ~5 sec smoke test

python -m scripts.run_cross_dataset
```

Writes `results/cross_dataset_transfer.csv` (84 long-format rows) and `results/cross_dataset_summary.csv`.

If you see the joblib `UserWarning` about `sklearn.utils.parallel.delayed` repeating many times: cosmetic, doesn't affect numbers. Silence with `$env:PYTHONWARNINGS="ignore::UserWarning"` (PowerShell) or `export PYTHONWARNINGS=ignore::UserWarning` (bash) before running.

Expected best-overall per pair:

| Source → Target | macro-F1 | Model | Norm | Threshold |
|---|---|---|---|---|
| campanella → nurse | 0.507 | RF | global | fixed_0.5 |
| campanella → wesad | 0.468 | XGB | global | fixed_0.5 |
| nurse → campanella | 0.474 | SVM | per_subject | source_tuned |
| nurse → wesad | 0.439 | XGB | global | fixed_0.5 |
| wesad → campanella | 0.393 | XGB | global | source_tuned |
| wesad → nurse | 0.376 | RF | per_subject | source_tuned |
| **wesad+campanella → nurse** | **0.526** | SVM | global | source_tuned |

All seeded with `random_state=42` — output is deterministic across reruns on the same machine.

## 7. Phase 6 — aggregation tables for the paper

```
python -m scripts.analyze_cross_dataset
# Optional: --markdown for paper-ready tables (needs `pip install tabulate`)
```

Writes six tables to `results/cross_dataset_analysis/`:
- `table1_macro_f1.csv` — wide-format macro-F1 per (source, target, norm, threshold)
- `table2_winning_models.csv` — which of rf/svm/xgb won each cell
- `table3_normalization_gap.csv` — per_subject minus global
- `table4_threshold_gap.csv` — source_tuned minus fixed_0.5
- `table5_best_overall.csv` — best-overall per (source, target)
- `table6_vs_phase5.csv` — Phase 6 best vs Phase 5 within-target pooled baseline

## Determinism and known sources of variance

All RNG is seeded with `random_state=42`. Results should reproduce bitwise on the same machine and Python/library versions. Cross-machine drift can come from:
- BLAS / OpenMP thread-count differences affecting floating-point reduction order
- scikit-learn / xgboost minor-version changes (we don't pin minor versions)
- joblib parallel workers — set `n_jobs=1` in `MODEL_FACTORIES` if you need strict bitwise reproducibility

For paper-grade numbers, run on a single machine and quote the library versions from `pip freeze > requirements_frozen.txt`.

## Troubleshooting

- **`No module named 'cvxopt'`** — `pip install cvxopt`. Required by neurokit2's cvxEDA. Pre-built wheels exist for Windows/Linux/macOS.
- **`features_unnormalized.csv not found`** during Phase 5 — `scripts/run_within_dataset.py` defaults to `data_extracted/features.csv` (the pre-Phase-6 name). Pass `--features-csv data_extracted/features_unnormalized.csv` explicitly.
- **Empty `data_extracted/nurse/`** — `scripts/extract_data.py` requires the Hosseini zip plus `SurveyResults.xlsx`. The XLSX must be present at extraction time, not added later.
- **Phase 6 fails with `No global feature file registered for source=...`** — one of the five feature files in step 4 is missing or named wrong. Check that the filename matches exactly what `FEATURE_FILES` in `scripts/run_cross_dataset.py` expects.
- **`UserWarning: sklearn.utils.parallel.delayed`** spam — cosmetic, see step 6 for how to silence.
