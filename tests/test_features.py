"""
Smoke test for src.features.

Runs the full pipeline on one subject per dataset:
    load_subject -> preprocess_subject -> create_windows -> extract_features

Prints feature counts, NaN counts per column, and value ranges so we can
sanity-check that nothing is silently broken.

Run from repo root:
    python -m tests.test_features
"""

import numpy as np
import pandas as pd

from src.data_loader import load_subject
from src.preprocessing import preprocess_subject
from src.windowing import create_windows
from src.labeling import get_campanella_labels, get_wesad_labels, get_nurse_labels
from src.features import extract_features


# One representative subject per dataset (matches test_preprocessing).
TEST_SUBJECTS = {
    "campanella": "subject_01",
    "wesad": "S2",
    "nurse": "83_1604630543",
}

DATA_ROOT = "data_extracted"


def _get_phases(dataset_name, subject_id, signals):
    """Dispatch to the right labeling function per dataset."""
    if dataset_name == "campanella":
        duration = min(
            (sig.index[-1] - sig.index[0]).total_seconds()
            for sig in signals.values() if len(sig) > 0
        )
        return get_campanella_labels(duration)
    if dataset_name == "wesad":
        return get_wesad_labels(subject_id)
    if dataset_name == "nurse":
        return get_nurse_labels(subject_id)
    raise ValueError(f"Unknown dataset: {dataset_name}")


def _summarize_features(df):
    """Print feature inventory, NaN counts, and value ranges."""
    meta_cols = [c for c in
                 ["dataset", "subject_id", "window_idx", "start_sec", "end_sec", "label"]
                 if c in df.columns]
    feat_cols = [c for c in df.columns if c not in meta_cols]

    print(f"  Shape: {df.shape} ({len(feat_cols)} features, {len(df)} windows)")
    print(f"  Label distribution: {df['label'].value_counts().to_dict()}")

    # NaN counts per column — anything > 0 worth flagging
    nan_counts = df[feat_cols].isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) == 0:
        print(f"  NaN check: ✓ no NaN in any feature column")
    else:
        print(f"  NaN check: {len(nan_cols)} columns have NaN")
        for col, n in nan_cols.items():
            pct = 100 * n / len(df)
            print(f"    {col:30s}  {n:4d}/{len(df)}  ({pct:.1f}%)")

    # Per-feature-family stats — one representative feature per group
    print(f"  Sample feature values (mean ± std across windows):")
    representatives = [
        "hrv_rmssd", "hrv_lf_hf_ratio",
        "eda_mean", "eda_scr_count", "eda_phasic_std",
        "bvp_spec_entropy", "bvp_dom_freq",
        "hr_mean",
        "acc_mag_mean", "acc_jerk_std",
        "temp_slope",
    ]
    for col in representatives:
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        if len(vals) == 0:
            print(f"    {col:25s}  all NaN")
            continue
        print(f"    {col:25s}  "
              f"mean={vals.mean():+.4f}  "
              f"std={vals.std():.4f}  "
              f"min={vals.min():+.4f}  "
              f"max={vals.max():+.4f}")


def _check_invariants(df, dataset_name):
    """Sanity-check the feature DataFrame for obvious problems."""
    issues = []

    # Expected feature count: ~40 (8 hrv + 10 eda + 6 bvp + 4 hr + 8 acc + 4 temp)
    meta_cols = [c for c in
                 ["dataset", "subject_id", "window_idx", "start_sec", "end_sec", "label"]
                 if c in df.columns]
    feat_cols = [c for c in df.columns if c not in meta_cols]
    if len(feat_cols) != 40:
        issues.append(f"Expected 40 features, got {len(feat_cols)}")

    # All windows should have a label of 0 or 1
    if not set(df["label"].unique()).issubset({0, 1}):
        issues.append(f"Unexpected labels: {df['label'].unique()}")

    # No infs anywhere (NaN is fine, inf is a bug)
    inf_cols = []
    for col in feat_cols:
        if np.any(np.isinf(df[col].values.astype(float))):
            inf_cols.append(col)
    if inf_cols:
        issues.append(f"Inf values in: {inf_cols}")

    # HR features should land in physiologically plausible ranges
    if "hr_mean" in df.columns:
        hr_vals = df["hr_mean"].dropna()
        if len(hr_vals) > 0 and (hr_vals.min() < 30 or hr_vals.max() > 200):
            issues.append(
                f"hr_mean out of range: [{hr_vals.min():.1f}, {hr_vals.max():.1f}]"
            )

    # HRV RMSSD typically 10-200 ms for healthy adults; flag if way outside
    if "hrv_rmssd" in df.columns:
        rmssd_vals = df["hrv_rmssd"].dropna()
        if len(rmssd_vals) > 0 and rmssd_vals.max() > 500:
            issues.append(
                f"hrv_rmssd suspiciously high: max={rmssd_vals.max():.1f} ms"
            )

    if issues:
        print(f"  ⚠ ISSUES:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print(f"  ✓ all invariants pass")


def test_features_one_subject(dataset_name, subject_id):
    print(f"\n{'='*72}")
    print(f"  {dataset_name.upper()} / {subject_id}")
    print(f"{'='*72}")

    # 1. Load
    signals_raw, sampling_rates = load_subject(
        dataset_name=dataset_name,
        subject_id=subject_id,
        data_root=DATA_ROOT,
    )
    if not signals_raw:
        print(f"  ⚠ No signals loaded")
        return

    # 2. Preprocess
    signals_clean = preprocess_subject(signals_raw, dataset_name=dataset_name)

    # 3. Phases + windowing
    phases = _get_phases(dataset_name, subject_id, signals_clean)
    if not phases:
        print(f"  ⚠ No phases for {subject_id}, skipping")
        return

    windows = create_windows(signals_clean, phases)
    print(f"  Windows created: {len(windows)}")
    if len(windows) == 0:
        print(f"  ⚠ Zero windows — skipping feature extraction")
        return

    # 4. Features
    df = extract_features(
        windows,
        subject_id=subject_id,
        dataset_name=dataset_name,
        sampling_rates=sampling_rates,
    )

    # 5. Summarize + sanity-check
    _summarize_features(df)
    _check_invariants(df, dataset_name)


def test_all():
    for dataset_name, subject_id in TEST_SUBJECTS.items():
        try:
            test_features_one_subject(dataset_name, subject_id)
        except Exception as e:
            print(f"\n  ✗ {dataset_name}/{subject_id} failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    test_all()
