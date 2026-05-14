"""
Run within-dataset LOSO baselines on all three datasets.

Reads `data_extracted/features.csv`, iterates over {campanella, wesad, nurse}
× {rf, xgb, svm}, runs leave-one-subject-out CV with optional inner
hyperparameter search and decision-threshold tuning, and writes:

    results/within_dataset_loso.csv      # one row per (dataset, model, held_out)
    results/within_dataset_summary.csv   # mean ± std per (dataset, model)

For Nurse, groups by nurse_id (`session_id.split("_")[0]`) — not session — so
sessions from the same nurse never appear on both sides of a fold split.

Usage:
    python -m scripts.run_within_dataset
    python -m scripts.run_within_dataset --no-tune --no-search   # fast pass
    python -m scripts.run_within_dataset --datasets wesad        # one dataset
    python -m scripts.run_within_dataset --models rf xgb         # subset of models
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

# Allow `python scripts/run_within_dataset.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import MODEL_FACTORIES, loso_evaluate, summarize_loso


FEATURES_CSV = Path("data_extracted/features.csv")
RESULTS_DIR = Path("results")
LOSO_OUT = RESULTS_DIR / "within_dataset_loso.csv"
SUMMARY_OUT = RESULTS_DIR / "within_dataset_summary.csv"

DATASETS = ["campanella", "wesad", "nurse"]


def add_nurse_id(df):
    """Add a `nurse_id` column for nurse rows (session_id is `{nurse}_{ts}`)."""
    df = df.copy()
    is_nurse = df["dataset"] == "nurse"
    df.loc[is_nurse, "nurse_id"] = df.loc[is_nurse, "subject_id"].str.split("_").str[0]
    return df


def run_one(dataset_df, dataset_name, models, do_search, do_tune, verbose):
    """Run LOSO for one dataset across the requested models."""
    group_col = "nurse_id" if dataset_name == "nurse" else "subject_id"
    n_groups = dataset_df[group_col].nunique()
    n_windows = len(dataset_df)

    print(f"\n{'='*72}")
    print(f"  {dataset_name.upper()}  ({n_groups} groups, {n_windows} windows, "
          f"group_col={group_col!r})")
    print(f"{'='*72}")

    all_rows = []
    for model_name in models:
        if model_name not in MODEL_FACTORIES:
            print(f"\n  [{model_name}] not available — skipping.")
            continue

        print(f"\n  --- {model_name} ---")
        t0 = time.time()
        results = loso_evaluate(
            dataset_df,
            model_name=model_name,
            group_col=group_col,
            do_hparam_search=do_search,
            do_threshold_tuning=do_tune,
            verbose=verbose,
        )
        elapsed = time.time() - t0

        if len(results) == 0:
            print(f"  No usable folds for {model_name} on {dataset_name}.")
            continue

        results.insert(0, "dataset", dataset_name)
        all_rows.append(results)

        print(
            f"  → {len(results)} folds in {elapsed:.1f}s. "
            f"Mean macro-F1: {results['macro_f1'].mean():.3f} "
            f"(± {results['macro_f1'].std():.3f})"
        )

    if not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features-csv", type=Path, default=FEATURES_CSV,
        help="Path to features.csv (default: data_extracted/features.csv)",
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=DATASETS,
        help="Subset of datasets to evaluate.",
    )
    parser.add_argument(
        "--models", nargs="+", default=list(MODEL_FACTORIES.keys()),
        help=f"Models to run. Available: {list(MODEL_FACTORIES.keys())}",
    )
    parser.add_argument(
        "--no-search", action="store_true",
        help="Skip inner hyperparameter search (use defaults). ~12x faster.",
    )
    parser.add_argument(
        "--no-tune", action="store_true",
        help="Skip threshold tuning (use 0.5).",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-fold output.",
    )
    args = parser.parse_args()

    if not args.features_csv.exists():
        print(f"ERROR: {args.features_csv} not found. Run scripts/extract_features.py first.")
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.features_csv}...")
    features = pd.read_csv(args.features_csv)
    features = add_nurse_id(features)
    print(f"  {len(features)} rows × {len(features.columns)} cols")
    print(f"  Datasets: {features['dataset'].value_counts().to_dict()}")

    # Warn on missing models so misspelled flags don't silently no-op.
    unknown = [m for m in args.models if m not in MODEL_FACTORIES]
    if unknown:
        print(f"\nWARNING: requested models not available: {unknown}")
        print(f"  Available: {list(MODEL_FACTORIES.keys())}")

    all_results = []
    for ds in args.datasets:
        ds_df = features[features["dataset"] == ds].reset_index(drop=True)
        if len(ds_df) == 0:
            print(f"\n  [{ds}] no rows in features.csv — skipping.")
            continue
        results = run_one(
            ds_df,
            dataset_name=ds,
            models=args.models,
            do_search=not args.no_search,
            do_tune=not args.no_tune,
            verbose=not args.quiet,
        )
        if len(results) > 0:
            all_results.append(results)

    if not all_results:
        print("\nNo results produced. Check the inputs above.")
        sys.exit(1)

    combined = pd.concat(all_results, ignore_index=True)

    # best_params is a dict — stringify for CSV roundtripping.
    if "best_params" in combined.columns:
        combined["best_params"] = combined["best_params"].astype(str)

    combined.to_csv(LOSO_OUT, index=False)
    print(f"\nWrote {LOSO_OUT} ({len(combined)} rows)")

    summary = summarize_loso(combined, by=("dataset", "model"))
    summary.to_csv(SUMMARY_OUT)
    print(f"Wrote {SUMMARY_OUT}")

    # Compact human-readable summary on stdout.
    print(f"\n{'='*72}")
    print("  Per-fold summary (macro-F1 mean ± std across folds)")
    print(f"{'='*72}")
    per_fold = combined[combined["held_out"] != "POOLED"]
    pivot = (
        per_fold.groupby(["dataset", "model"])["macro_f1"]
        .agg(["mean", "std", "count"])
        .round(3)
    )
    print(pivot.to_string())

    # Pooled metrics — informative when many per-fold values are degenerate
    # (single-class held-out subjects).
    pooled = combined[combined["held_out"] == "POOLED"]
    if len(pooled) > 0:
        print(f"\n{'='*72}")
        print("  Pooled-across-folds metrics (concatenated predictions)")
        print(f"{'='*72}")
        pooled_view = (
            pooled.set_index(["dataset", "model"])
            [["macro_f1", "balanced_accuracy", "accuracy", "f1_0", "f1_1"]]
            .round(3)
        )
        print(pooled_view.to_string())


if __name__ == "__main__":
    main()
