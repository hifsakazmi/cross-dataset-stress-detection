"""
Phase 6: cross-dataset transfer experiments.

Runs every (source, target, model, normalization, threshold_policy) combo:

  Pairwise: 6 pairs × 3 models × 2 norms × 2 thr = 72 experiments
  Multi-source (WESAD+Campanella → Nurse): 1 × 3 × 2 × 2 = 12 experiments
  Total: 84 experiments

For each, trains on the entire source dataset's windows and predicts on the
entire target dataset's windows. No retraining; the only "domain adaptation"
is the choice of signal-level normalization (per-subject vs global, fit on
source). Threshold policy controls whether the 0.5 default or a source-tuned
optimum is used at decision time.

Reads five feature CSVs produced by scripts/extract_features.py (Phase 6):
    features_per_subject.csv
    features_global_src_<dataset>.csv  (one per source, plus a multi-source one)

Writes:
    results/cross_dataset_transfer.csv  — long-format, one row per experiment
    results/cross_dataset_summary.csv   — same data aggregated per (source,target)

Run from repo root:
    python -m scripts.run_cross_dataset
    python -m scripts.run_cross_dataset --no-search          # skip hparam grid
    python -m scripts.run_cross_dataset --models rf svm      # subset
    python -m scripts.run_cross_dataset --pairs wesad-campanella  # one pair
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.models import MODEL_FACTORIES, evaluate_transfer


# --- paths ---------------------------------------------------------------

DATA_DIR = Path("data_extracted")
RESULTS_DIR = Path("results")

FEATURE_FILES = {
    # (normalization, source_key) -> path. source_key is "any" for per_subject
    # (one file covers all sources), else the source dataset name(s).
    ("per_subject", "any"): DATA_DIR / "features_per_subject.csv",
    ("global", "wesad"): DATA_DIR / "features_global_src_wesad.csv",
    ("global", "campanella"): DATA_DIR / "features_global_src_campanella.csv",
    ("global", "nurse"): DATA_DIR / "features_global_src_nurse.csv",
    ("global", "wesad+campanella"):
        DATA_DIR / "features_global_src_wesad_campanella.csv",
}

# --- experiments ---------------------------------------------------------

PAIRWISE = [
    ("wesad", "campanella"),
    ("wesad", "nurse"),
    ("campanella", "wesad"),
    ("campanella", "nurse"),
    ("nurse", "wesad"),
    ("nurse", "campanella"),
]

MULTI_SOURCE = [
    # (source_components, target). Source components are unioned by row.
    (("wesad", "campanella"), "nurse"),
]

NORMALIZATIONS = ["per_subject", "global"]
THRESHOLD_POLICIES = [
    ("fixed_0.5", False),       # do_threshold_tuning=False  → 0.5
    ("source_tuned", True),     # do_threshold_tuning=True   → tuned on src slice
]


# --- helpers -------------------------------------------------------------

def add_nurse_id(df):
    """Mirror scripts/run_within_dataset.add_nurse_id: nurse_id = session.split('_')[0].

    Required as the inner-CV group column when Nurse is on the source side.
    For non-Nurse rows, nurse_id is set to subject_id so the column is always
    populated (avoids NaN groups in inner GroupKFold).
    """
    out = df.copy()
    is_nurse = out["dataset"] == "nurse"
    out["nurse_id"] = out["subject_id"].astype(str)
    out.loc[is_nurse, "nurse_id"] = (
        out.loc[is_nurse, "subject_id"].astype(str).str.split("_").str[0]
    )
    return out


def _source_key(source):
    """Normalize a source spec ('wesad' or ('wesad','campanella')) to a feature-file key."""
    if isinstance(source, str):
        return source
    return "+".join(source)


def _source_row_mask(df, source):
    """Boolean mask over df rows that belong to `source` (str or tuple)."""
    if isinstance(source, str):
        return df["dataset"] == source
    return df["dataset"].isin(source)


def _group_col_for_source(source):
    """Inner-CV grouping: nurse_id when Nurse is in the source pool, else subject_id."""
    if isinstance(source, str):
        return "nurse_id" if source == "nurse" else "subject_id"
    return "nurse_id" if "nurse" in source else "subject_id"


def _load_features(normalization, source):
    """Load the appropriate feature CSV for (normalization, source)."""
    if normalization == "per_subject":
        path = FEATURE_FILES[("per_subject", "any")]
    elif normalization == "global":
        key = _source_key(source)
        path = FEATURE_FILES.get(("global", key))
        if path is None:
            raise FileNotFoundError(
                f"No global feature file registered for source={source!r}. "
                f"Expected key 'global'/{key!r}. Available: "
                f"{[k for k in FEATURE_FILES if k[0]=='global']}"
            )
    else:
        raise ValueError(f"Unknown normalization: {normalization}")

    if not path.exists():
        raise FileNotFoundError(
            f"Missing feature file: {path}. Run scripts/extract_features.py "
            f"with the appropriate --normalize-mode / --source-datasets flags."
        )
    df = pd.read_csv(path)
    df = add_nurse_id(df)
    return df


def _format_source(source):
    if isinstance(source, str):
        return source
    return "+".join(source)


# --- main loop -----------------------------------------------------------

def run_one_experiment(
    features_df, source, target, model_name, normalization,
    threshold_label, do_threshold_tuning,
    do_hparam_search=True,
    verbose=True,
):
    """Run a single (source, target, model, norm, threshold) experiment."""
    src_mask = _source_row_mask(features_df, source)
    tgt_mask = features_df["dataset"] == target

    train_df = features_df[src_mask].reset_index(drop=True)
    test_df = features_df[tgt_mask].reset_index(drop=True)

    if len(train_df) == 0:
        raise ValueError(f"No source rows found for {source!r}")
    if len(test_df) == 0:
        raise ValueError(f"No target rows found for {target!r}")

    group_col = _group_col_for_source(source)

    t0 = time.time()
    result = evaluate_transfer(
        train_df=train_df,
        test_df=test_df,
        model_name=model_name,
        group_col=group_col,
        do_hparam_search=do_hparam_search,
        do_threshold_tuning=do_threshold_tuning,
    )
    elapsed = time.time() - t0

    # Annotate with experiment keys.
    row = {
        "source": _format_source(source),
        "target": target,
        "model": model_name,
        "normalization": normalization,
        "threshold_policy": threshold_label,
        "elapsed_sec": round(elapsed, 2),
    }
    row.update(result)

    if verbose:
        print(
            f"  [{row['source']:>20s} → {target:<11s}] "
            f"{model_name:<4s} norm={normalization:<11s} thr={threshold_label:<12s} "
            f"macro_f1={row.get('macro_f1', float('nan')):.3f} "
            f"bal_acc={row.get('balanced_accuracy', float('nan')):.3f} "
            f"thr={row['best_threshold']:.2f} "
            f"({elapsed:.1f}s)"
        )
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", default=list(MODEL_FACTORIES.keys()),
        choices=list(MODEL_FACTORIES.keys()),
        help="Subset of models to run.",
    )
    parser.add_argument(
        "--normalizations", nargs="+", default=NORMALIZATIONS,
        choices=NORMALIZATIONS,
        help="Which normalization arms to run.",
    )
    parser.add_argument(
        "--threshold-policies", nargs="+",
        default=[label for label, _ in THRESHOLD_POLICIES],
        choices=[label for label, _ in THRESHOLD_POLICIES],
        help="Which threshold policies to run.",
    )
    parser.add_argument(
        "--pairs", nargs="+", default=None,
        help="Restrict to specific pairwise experiments, formatted as "
             "'source-target' (e.g. 'wesad-nurse'). Default: all 6 pairs.",
    )
    parser.add_argument(
        "--skip-multi-source", action="store_true",
        help="Skip the WESAD+Campanella → Nurse multi-source experiment.",
    )
    parser.add_argument(
        "--no-search", action="store_true",
        help="Skip inner hparam search (use defaults). ~10x faster.",
    )
    parser.add_argument(
        "--output", type=Path, default=RESULTS_DIR / "cross_dataset_transfer.csv",
        help="Long-format results CSV.",
    )
    parser.add_argument(
        "--summary-output", type=Path,
        default=RESULTS_DIR / "cross_dataset_summary.csv",
        help="Aggregated per-(source,target) summary CSV.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-experiment progress lines.",
    )
    args = parser.parse_args()

    verbose = not args.quiet
    do_hparam_search = not args.no_search

    # Resolve pairs filter.
    pairs = PAIRWISE
    if args.pairs:
        wanted = set()
        for p in args.pairs:
            try:
                s, t = p.split("-")
            except ValueError:
                parser.error(f"--pairs entry not in 'source-target' form: {p!r}")
            wanted.add((s, t))
        pairs = [p for p in PAIRWISE if p in wanted]
        if not pairs:
            parser.error(f"No PAIRWISE entries match --pairs {args.pairs!r}")

    threshold_policies = [
        (label, do_tune) for label, do_tune in THRESHOLD_POLICIES
        if label in args.threshold_policies
    ]

    # Resolve experiments to run.
    experiments = []
    for source, target in pairs:
        for norm in args.normalizations:
            for thr_label, do_tune in threshold_policies:
                for model in args.models:
                    experiments.append(
                        (source, target, model, norm, thr_label, do_tune)
                    )
    if not args.skip_multi_source:
        for source_tuple, target in MULTI_SOURCE:
            for norm in args.normalizations:
                for thr_label, do_tune in threshold_policies:
                    for model in args.models:
                        experiments.append(
                            (source_tuple, target, model, norm, thr_label, do_tune)
                        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Phase 6 cross-dataset transfer: {len(experiments)} experiments")
    print(f"  models:               {args.models}")
    print(f"  normalizations:       {args.normalizations}")
    print(f"  threshold policies:   {[t for t, _ in threshold_policies]}")
    print(f"  pairs:                {pairs}")
    print(f"  multi-source skipped: {args.skip_multi_source}")
    print(f"  hparam search:        {do_hparam_search}")
    print()

    # Cache loaded feature files since loading is the slowest part.
    feature_cache = {}
    def _get_features(normalization, source):
        if normalization == "per_subject":
            key = ("per_subject", "any")
        else:
            key = ("global", _source_key(source))
        if key not in feature_cache:
            feature_cache[key] = _load_features(normalization, source)
            if verbose:
                df = feature_cache[key]
                print(f"  loaded {FEATURE_FILES[key].name}: "
                      f"{len(df)} rows, datasets={df['dataset'].value_counts().to_dict()}")
        return feature_cache[key]

    rows = []
    failures = []
    t_start = time.time()
    for i, (source, target, model, norm, thr_label, do_tune) in enumerate(experiments, 1):
        try:
            features = _get_features(norm, source)
            row = run_one_experiment(
                features_df=features,
                source=source, target=target,
                model_name=model,
                normalization=norm,
                threshold_label=thr_label,
                do_threshold_tuning=do_tune,
                do_hparam_search=do_hparam_search,
                verbose=verbose,
            )
            rows.append(row)
        except Exception as e:
            msg = f"{_format_source(source)} → {target} / {model} / {norm} / {thr_label}: {e}"
            print(f"  [ERROR] {msg}")
            import traceback; traceback.print_exc()
            failures.append(msg)

        if verbose and i % 12 == 0:
            print(f"  --- progress {i}/{len(experiments)}, "
                  f"elapsed {time.time()-t_start:.0f}s ---")

    total_elapsed = time.time() - t_start
    print(f"\nDone in {total_elapsed:.1f}s. "
          f"{len(rows)} succeeded, {len(failures)} failed.")

    if not rows:
        print("No successful experiments. Nothing to save.")
        sys.exit(1)

    results_df = pd.DataFrame(rows)
    results_df.to_csv(args.output, index=False)
    print(f"  wrote {args.output} ({len(results_df)} rows)")

    # Aggregated summary: mean over (source, target, normalization, threshold_policy)
    # — useful for the paper table where the model dimension is collapsed or
    # broken out separately. Here we keep model on the index and just sort.
    summary_cols = [
        "source", "target", "model", "normalization", "threshold_policy",
        "n_train", "n_test", "best_threshold",
        "macro_f1", "balanced_accuracy", "accuracy", "roc_auc",
        "precision_0", "recall_0", "f1_0",
        "precision_1", "recall_1", "f1_1",
    ]
    keep = [c for c in summary_cols if c in results_df.columns]
    summary = results_df[keep].sort_values(
        ["source", "target", "model", "normalization", "threshold_policy"]
    )
    summary.to_csv(args.summary_output, index=False)
    print(f"  wrote {args.summary_output} ({len(summary)} rows)")

    if failures:
        print(f"\n  {len(failures)} failures:")
        for f in failures:
            print(f"    - {f}")


if __name__ == "__main__":
    main()
