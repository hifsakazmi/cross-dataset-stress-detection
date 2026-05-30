"""
Phase 6 results analysis.

Reads results/cross_dataset_transfer.csv and produces the six aggregation
tables used in the paper writeup:

  Table 1: macro-F1 per (source, target, normalization, threshold_policy),
           best-of-three-models per cell.
  Table 2: which model (rf/svm/xgb) won each cell of Table 1.
  Table 3: normalization gap — per_subject minus global, best-model per cell.
  Table 4: threshold gap — source_tuned minus fixed_0.5, best-model per cell.
  Table 5: best-overall per (source, target) — any model, any norm, any thr.
  Table 6: Phase 6 best vs Phase 5 within-target pooled baseline.

Optional --markdown flag emits paper-ready Markdown tables in addition to
the plain-text console summary. Saves the same six tables as CSVs under
results/cross_dataset_analysis/ for downstream notebook use.

Run from repo root:
    python -m scripts.analyze_cross_dataset
    python -m scripts.analyze_cross_dataset --markdown
    python -m scripts.analyze_cross_dataset --transfer-csv path/to/other.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# Phase 5 within-dataset pooled macro-F1 baselines (from project_context.md
# Phase 5 results). Used in Table 6 as the within-target reference column.
# If these numbers are ever rerun, update here.
PHASE5_WITHIN_POOLED = {
    "campanella": 0.684,  # RF
    "wesad":      0.758,  # SVM
    "nurse":      0.439,  # SVM
}

DEFAULT_INPUT = Path("results/cross_dataset_transfer.csv")
DEFAULT_OUTDIR = Path("results/cross_dataset_analysis")

# Column ordering used in the wide tables so paper rows always read
# (global,fixed) (global,tuned) (per_subject,fixed) (per_subject,tuned).
NORM_ORDER = ["global", "per_subject"]
THR_ORDER = ["fixed_0.5", "source_tuned"]
WIDE_COL_ORDER = [(n, t) for n in NORM_ORDER for t in THR_ORDER]


# --- table builders ------------------------------------------------------

def _best_per_cell(df, cell_cols):
    """For each unique combination of cell_cols, pick the row with the
    highest macro_f1. Returns a DataFrame with the same columns as df."""
    idx = df.groupby(cell_cols, observed=True)["macro_f1"].idxmax()
    return df.loc[idx].reset_index(drop=True)


def table1_macro_f1(df):
    """Best-model-per-cell macro-F1, wide-format by (norm, threshold)."""
    best = _best_per_cell(
        df, ["source", "target", "normalization", "threshold_policy"]
    )
    pivot = best.pivot_table(
        index=["source", "target"],
        columns=["normalization", "threshold_policy"],
        values="macro_f1",
        aggfunc="first",
    )
    # Ensure consistent column order even if some experiments are missing.
    cols = [c for c in WIDE_COL_ORDER if c in pivot.columns]
    pivot = pivot[cols].round(3)
    return pivot


def table2_winning_models(df):
    """Same shape as Table 1 but values are which model (rf/svm/xgb) won."""
    best = _best_per_cell(
        df, ["source", "target", "normalization", "threshold_policy"]
    )
    pivot = best.pivot_table(
        index=["source", "target"],
        columns=["normalization", "threshold_policy"],
        values="model",
        aggfunc="first",
    )
    cols = [c for c in WIDE_COL_ORDER if c in pivot.columns]
    return pivot[cols]


def table3_norm_gap(df):
    """per_subject minus global, holding (source, target, threshold) fixed.
    Per cell, take the best model under each normalization."""
    best = _best_per_cell(
        df, ["source", "target", "normalization", "threshold_policy"]
    )
    wide = best.set_index(
        ["source", "target", "threshold_policy", "normalization"]
    )["macro_f1"].unstack("normalization")
    wide["delta_per_subject_vs_global"] = (
        wide["per_subject"] - wide["global"]
    )
    return wide.round(3)


def table4_threshold_gap(df):
    """source_tuned minus fixed_0.5, holding (source, target, norm) fixed."""
    best = _best_per_cell(
        df, ["source", "target", "normalization", "threshold_policy"]
    )
    wide = best.set_index(
        ["source", "target", "normalization", "threshold_policy"]
    )["macro_f1"].unstack("threshold_policy")
    wide["delta_tuned_vs_fixed"] = wide["source_tuned"] - wide["fixed_0.5"]
    return wide.round(3)


def table5_best_overall(df):
    """Single best row per (source, target) across all knobs."""
    idx = df.groupby(["source", "target"])["macro_f1"].idxmax()
    cols = [
        "source", "target", "model", "normalization", "threshold_policy",
        "macro_f1", "balanced_accuracy", "accuracy", "roc_auc",
        "best_threshold", "n_train", "n_test",
    ]
    return df.loc[idx, cols].sort_values(["source", "target"]).round(3).reset_index(drop=True)


def table6_vs_phase5(best_overall):
    """Best-overall per pair compared to Phase 5 within-target baseline."""
    rows = []
    for _, r in best_overall.iterrows():
        within = PHASE5_WITHIN_POOLED.get(r["target"])
        rows.append({
            "source": r["source"],
            "target": r["target"],
            "phase6_best_macro_f1": round(float(r["macro_f1"]), 3),
            "phase5_within_target_pooled": within,
            "delta_vs_within_target": round(float(r["macro_f1"]) - within, 3)
                                       if within is not None else np.nan,
            "model": r["model"],
            "normalization": r["normalization"],
            "threshold_policy": r["threshold_policy"],
        })
    return pd.DataFrame(rows)


# --- formatting ----------------------------------------------------------

def _print_header(title):
    print("\n" + "=" * 100)
    print(f"  {title}")
    print("=" * 100)


def _to_markdown(df, **kwargs):
    """Wrap pandas to_markdown; emit fallback if tabulate isn't installed."""
    try:
        return df.to_markdown(**kwargs)
    except ImportError:
        return df.to_string(**kwargs)


# --- main ----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transfer-csv", type=Path, default=DEFAULT_INPUT,
        help=f"Long-format transfer results (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--outdir", type=Path, default=DEFAULT_OUTDIR,
        help=f"Directory for per-table CSVs (default: {DEFAULT_OUTDIR})",
    )
    parser.add_argument(
        "--markdown", action="store_true",
        help="Also print paper-ready Markdown tables.",
    )
    args = parser.parse_args()

    if not args.transfer_csv.exists():
        raise SystemExit(
            f"ERROR: {args.transfer_csv} not found. Run "
            f"scripts/run_cross_dataset.py first."
        )

    df = pd.read_csv(args.transfer_csv)
    print(f"Loaded {len(df)} rows from {args.transfer_csv}")
    print(f"  sources: {sorted(df['source'].unique())}")
    print(f"  targets: {sorted(df['target'].unique())}")
    print(f"  models:  {sorted(df['model'].unique())}")

    args.outdir.mkdir(parents=True, exist_ok=True)

    # --- Table 1 -------------------------------------------------------
    t1 = table1_macro_f1(df)
    _print_header("Table 1: macro-F1 per cell (best-of-three models)")
    print(t1.to_string())
    t1.to_csv(args.outdir / "table1_macro_f1.csv")
    if args.markdown:
        print("\n" + _to_markdown(t1))

    # --- Table 2 -------------------------------------------------------
    t2 = table2_winning_models(df)
    _print_header("Table 2: which model won each cell")
    print(t2.to_string())
    t2.to_csv(args.outdir / "table2_winning_models.csv")
    if args.markdown:
        print("\n" + _to_markdown(t2))

    # --- Table 3 -------------------------------------------------------
    t3 = table3_norm_gap(df)
    _print_header("Table 3: normalization gap (per_subject minus global)")
    print(t3.to_string())
    t3.to_csv(args.outdir / "table3_normalization_gap.csv")
    if args.markdown:
        print("\n" + _to_markdown(t3))

    # --- Table 4 -------------------------------------------------------
    t4 = table4_threshold_gap(df)
    _print_header("Table 4: threshold gap (source_tuned minus fixed_0.5)")
    print(t4.to_string())
    t4.to_csv(args.outdir / "table4_threshold_gap.csv")
    if args.markdown:
        print("\n" + _to_markdown(t4))

    # --- Table 5 -------------------------------------------------------
    t5 = table5_best_overall(df)
    _print_header("Table 5: best-overall per (source, target)")
    print(t5.to_string(index=False))
    t5.to_csv(args.outdir / "table5_best_overall.csv", index=False)
    if args.markdown:
        print("\n" + _to_markdown(t5, index=False))

    # --- Table 6 -------------------------------------------------------
    t6 = table6_vs_phase5(t5)
    _print_header("Table 6: Phase 6 best vs Phase 5 within-target baseline")
    print(t6.to_string(index=False))
    t6.to_csv(args.outdir / "table6_vs_phase5.csv", index=False)
    if args.markdown:
        print("\n" + _to_markdown(t6, index=False))

    print(f"\nAll six tables saved under {args.outdir}/")


if __name__ == "__main__":
    main()
