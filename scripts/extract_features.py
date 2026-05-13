"""
Full-cohort feature extraction.

Walks every subject in every dataset, runs:
    load_subject -> preprocess_subject -> create_windows -> extract_features

Concatenates all per-subject DataFrames into one big features table, saves to
data_extracted/features.parquet, and prints a summary so we can spot
subject-specific edge cases the smoke test missed.

Mirrors the iteration patterns in tests/test_windowing.py and src/eda.py so
exclusions (under-32-min Campanella subjects, empty-IBI nurses, unlabeled
nurse sessions) are handled consistently with Phase 2.

Run from repo root:
    python -m scripts.extract_features
    python -m scripts.extract_features --output data_extracted/features.csv
    python -m scripts.extract_features --datasets campanella wesad
"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import load_subject, list_subjects
from src.preprocessing import preprocess_subject
from src.windowing import create_windows
from src.labeling import get_campanella_labels, get_wesad_labels, get_nurse_labels
from src.features import extract_features


DATA_ROOT = "data_extracted"
DEFAULT_OUTPUT = "data_extracted/features.parquet"
ALL_DATASETS = ["campanella", "wesad", "nurse"]


# ---------- Per-dataset phase loading ----------

def _phases_campanella(subject_id, signals):
    """Compute Campanella phases from protocol. Raises ValueError if recording <32min."""
    duration = min(
        (sig.index[-1] - sig.index[0]).total_seconds()
        for sig in signals.values() if len(sig) > 0
    )
    return get_campanella_labels(duration)


def _phases_wesad(subject_id, signals):
    return get_wesad_labels(subject_id)


def _phases_nurse(session_id, signals):
    """Nurse phases may be empty (most sessions have no matching surveys)."""
    return get_nurse_labels(session_id)


PHASE_LOADERS = {
    "campanella": _phases_campanella,
    "wesad": _phases_wesad,
    "nurse": _phases_nurse,
}


# ---------- Process one subject ----------

def process_subject(dataset_name, subject_id, data_root=DATA_ROOT):
    """
    Returns (features_df, status_dict). features_df is None on skip/error.
    status_dict records what happened for the summary.
    """
    status = {
        "dataset": dataset_name,
        "subject_id": subject_id,
        "loaded": False,
        "n_signals": 0,
        "duration_min": 0.0,
        "n_phases": 0,
        "n_windows": 0,
        "skipped": False,
        "skip_reason": None,
        "error": None,
    }

    # 1. Load
    try:
        signals, sampling_rates = load_subject(
            dataset_name=dataset_name,
            subject_id=subject_id,
            data_root=data_root,
        )
    except Exception as e:
        status["error"] = f"load failed: {e}"
        return None, status

    if not signals:
        status["skipped"] = True
        status["skip_reason"] = "no signals loaded"
        return None, status

    status["loaded"] = True
    status["n_signals"] = len(signals)

    # 2. Preprocess
    try:
        signals = preprocess_subject(signals, dataset_name=dataset_name)
    except Exception as e:
        status["error"] = f"preprocessing failed: {e}"
        return None, status

    # 3. Phases
    try:
        phases = PHASE_LOADERS[dataset_name](subject_id, signals)
    except ValueError as e:
        # Campanella under-32-min subjects raise this
        status["skipped"] = True
        status["skip_reason"] = f"phase computation: {e}"
        return None, status
    except Exception as e:
        status["error"] = f"phase loading failed: {e}"
        return None, status

    if not phases:
        status["skipped"] = True
        status["skip_reason"] = "no labeled phases"
        return None, status

    status["n_phases"] = len(phases)

    # 4. Window
    try:
        windows = create_windows(signals, phases)
    except Exception as e:
        status["error"] = f"windowing failed: {e}"
        return None, status

    if not windows:
        status["skipped"] = True
        status["skip_reason"] = "zero windows"
        return None, status

    duration = min(
        (sig.index[-1] - sig.index[0]).total_seconds()
        for sig in signals.values() if len(sig) > 0
    )
    status["duration_min"] = duration / 60.0
    status["n_windows"] = len(windows)

    # 5. Features
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # neurokit2 / scipy edge-case warnings
            df = extract_features(
                windows,
                subject_id=subject_id,
                dataset_name=dataset_name,
                sampling_rates=sampling_rates,
            )
    except Exception as e:
        status["error"] = f"feature extraction failed: {e}"
        return None, status

    return df, status


# ---------- Per-dataset driver ----------

def process_dataset(dataset_name, data_root=DATA_ROOT):
    """Process every subject in a dataset. Returns (features_df, [statuses])."""
    print(f"\n{'='*72}")
    print(f"  {dataset_name.upper()}")
    print(f"{'='*72}")

    subjects = list_subjects(dataset_name, data_root=data_root)
    print(f"  Found {len(subjects)} subjects/sessions in {dataset_name}")

    all_dfs = []
    statuses = []
    t0 = time.time()

    for i, subject_id in enumerate(subjects, 1):
        df, status = process_subject(dataset_name, subject_id, data_root)
        statuses.append(status)

        # One-line per-subject log
        if status["error"]:
            print(f"  [{i:3d}/{len(subjects)}] ✗ {subject_id}: ERROR — {status['error']}")
        elif status["skipped"]:
            print(f"  [{i:3d}/{len(subjects)}] - {subject_id}: skipped ({status['skip_reason']})")
        else:
            print(
                f"  [{i:3d}/{len(subjects)}] ✓ {subject_id}: "
                f"{status['duration_min']:.1f} min, "
                f"{status['n_windows']} windows"
            )

        if df is not None:
            all_dfs.append(df)

    elapsed = time.time() - t0
    print(f"\n  {dataset_name}: {elapsed:.1f} sec for {len(subjects)} subjects "
          f"({elapsed / max(len(subjects), 1):.2f} sec/subject)")

    if not all_dfs:
        return pd.DataFrame(), statuses

    return pd.concat(all_dfs, ignore_index=True), statuses


# ---------- Summary ----------

def print_summary(features_df, all_statuses):
    """Print dataset-level rollup + NaN map + subject-level NaN flags."""
    print(f"\n{'='*72}")
    print(f"  SUMMARY")
    print(f"{'='*72}")

    # Dataset rollup
    print(f"\n  Per-dataset breakdown:")
    print(f"  {'dataset':12s}  {'subjects':>10s}  {'kept':>5s}  {'skipped':>8s}  "
          f"{'errors':>7s}  {'windows':>8s}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*5}  {'-'*8}  {'-'*7}  {'-'*8}")
    for ds in ALL_DATASETS:
        ds_statuses = [s for s in all_statuses if s["dataset"] == ds]
        n_total = len(ds_statuses)
        n_kept = sum(1 for s in ds_statuses if s["n_windows"] > 0 and not s["error"])
        n_skipped = sum(1 for s in ds_statuses if s["skipped"])
        n_errors = sum(1 for s in ds_statuses if s["error"])
        n_windows = sum(s["n_windows"] for s in ds_statuses)
        print(f"  {ds:12s}  {n_total:>10d}  {n_kept:>5d}  {n_skipped:>8d}  "
              f"{n_errors:>7d}  {n_windows:>8d}")

    if features_df.empty:
        print(f"\n  ⚠ No features extracted.")
        return

    # Total
    print(f"\n  Total feature rows: {len(features_df)}")
    print(f"  Feature columns: {features_df.shape[1]}")

    # Label balance per dataset
    print(f"\n  Label distribution:")
    for ds in ALL_DATASETS:
        sub = features_df[features_df["dataset"] == ds]
        if sub.empty:
            continue
        counts = sub["label"].value_counts().to_dict()
        n_stress = counts.get(1, 0)
        n_non = counts.get(0, 0)
        ratio = n_stress / n_non if n_non > 0 else float("inf")
        print(f"    {ds:12s}  stress={n_stress:4d}  non-stress={n_non:4d}  "
              f"ratio={ratio:.2f}")

    # NaN map per dataset
    meta_cols = ["dataset", "subject_id", "window_idx", "start_sec", "end_sec", "label"]
    feat_cols = [c for c in features_df.columns if c not in meta_cols]

    print(f"\n  NaN rate per feature (overall):")
    nan_rates = features_df[feat_cols].isna().mean().sort_values(ascending=False)
    flagged = nan_rates[nan_rates > 0]
    if len(flagged) == 0:
        print(f"    ✓ no NaN in any feature column")
    else:
        for col, rate in flagged.items():
            n = int(features_df[col].isna().sum())
            print(f"    {col:30s}  {n:5d}/{len(features_df)}  ({rate*100:.1f}%)")

    # Subject-level NaN flags — any subject with >50% NaN on a feature is suspicious
    print(f"\n  Subjects with >50% NaN on any feature (flag for review):")
    flagged_any = False
    for ds in ALL_DATASETS:
        sub = features_df[features_df["dataset"] == ds]
        for subject_id in sub["subject_id"].unique():
            sub_rows = sub[sub["subject_id"] == subject_id]
            subj_nan = sub_rows[feat_cols].isna().mean()
            high = subj_nan[subj_nan > 0.5]
            if len(high) > 0:
                flagged_any = True
                features_list = ", ".join(high.index.tolist()[:3])
                more = f" (+{len(high)-3} more)" if len(high) > 3 else ""
                print(f"    {ds}/{subject_id}  ({len(sub_rows)} windows): "
                      f"{features_list}{more}")
    if not flagged_any:
        print(f"    ✓ no subjects with extreme NaN rates")

    # Skipped/errored subjects worth knowing about
    skipped = [s for s in all_statuses if s["skipped"]]
    errored = [s for s in all_statuses if s["error"]]
    if skipped:
        print(f"\n  Skipped subjects ({len(skipped)}):")
        for s in skipped[:20]:  # cap output
            print(f"    {s['dataset']}/{s['subject_id']}: {s['skip_reason']}")
        if len(skipped) > 20:
            print(f"    ... +{len(skipped)-20} more")
    if errored:
        print(f"\n  ⚠ Errored subjects ({len(errored)}):")
        for s in errored:
            print(f"    {s['dataset']}/{s['subject_id']}: {s['error']}")


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="Full-cohort feature extraction")
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"Output path (.parquet or .csv). Default: {DEFAULT_OUTPUT}"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=ALL_DATASETS, choices=ALL_DATASETS,
        help="Which datasets to process. Default: all three.",
    )
    parser.add_argument(
        "--data-root", default=DATA_ROOT,
        help=f"Root of extracted data. Default: {DATA_ROOT}"
    )
    args = parser.parse_args()

    all_dfs = []
    all_statuses = []
    total_t0 = time.time()

    for dataset_name in args.datasets:
        df, statuses = process_dataset(dataset_name, data_root=args.data_root)
        if not df.empty:
            all_dfs.append(df)
        all_statuses.extend(statuses)

    total_elapsed = time.time() - total_t0
    print(f"\n  Total time: {total_elapsed:.1f} sec ({total_elapsed/60:.1f} min)")

    if not all_dfs:
        print(f"\n  ✗ No features extracted. Nothing to save.")
        return

    features_df = pd.concat(all_dfs, ignore_index=True)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".parquet":
        try:
            features_df.to_parquet(output_path, index=False)
        except ImportError:
            print(f"\n  ⚠ Parquet requires pyarrow or fastparquet. Falling back to CSV.")
            output_path = output_path.with_suffix(".csv")
            features_df.to_csv(output_path, index=False)
    elif output_path.suffix == ".csv":
        features_df.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unknown output format: {output_path.suffix}")

    print(f"\n  ✓ Saved {len(features_df)} rows to {output_path}")
    print(f"    ({features_df.shape[1]} columns, "
          f"{output_path.stat().st_size / (1024**2):.2f} MB)")

    print_summary(features_df, all_statuses)


if __name__ == "__main__":
    main()
