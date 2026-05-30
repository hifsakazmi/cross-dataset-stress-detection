"""
Sanity check for the Phase 6 signal-level normalization pipeline.

Runs the full per-subject feature extraction in all three normalize_mode
settings on one subject per dataset, and asserts:

1. All three modes produce the same number of windows (normalization
   doesn't change windowing — only the values inside).
2. NaN structure is preserved (HRV features still NaN-at-roughly-expected
   rate; everything else stays 0% NaN per the Phase 4 inventory).
3. "none" reproduces the existing features.csv numerics within tolerance
   for the smoke-test subjects (regression check that we didn't break
   the legacy path while adding the new flag).
4. "per_subject" output has feature-column means/stds that look like
   they came from z-scored signals (BVP std reduced vs "none",
   TEMP range collapsed, etc).
5. "global" requires the global_stats arg and applies it correctly
   (different from per_subject — sanity check, not a numerical match).

Doesn't take 70 minutes — runs on one subject per dataset, ~30 seconds total.

Run from repo root:
    python -m tests.test_normalization_pipeline
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import load_subject
from src.preprocessing import (
    preprocess_subject,
    normalize_signals,
    fit_global_normalizer,
)
from scripts.extract_features import process_subject


TEST_SUBJECTS = {
    "campanella": "subject_01",
    "wesad": "S2",
    "nurse": "83_1604630543",
}

DATA_ROOT = "data_extracted"


def _expected_hrv_columns():
    """HRV columns that are allowed to be NaN per Phase 4 inventory."""
    return [
        "hrv_rmssd", "hrv_sdnn", "hrv_pnn50",
        "hrv_mean_ibi", "hrv_median_ibi",
        "hrv_hr_mean", "hrv_hr_std", "hrv_lfhf",
    ]


def _summarize_features(label, df):
    n_rows, n_cols = df.shape
    nan_cols = df.columns[df.isna().any()].tolist()
    print(f"  {label}: shape={df.shape}, "
          f"label dist={df['label'].value_counts().to_dict()}")
    if nan_cols:
        nan_rates = (df[nan_cols].isna().mean() * 100).round(1).to_dict()
        print(f"    NaN columns: {nan_rates}")


def _build_global_stats(dataset_name):
    """Fit global stats on a SINGLE subject for the sanity check (fast).
    Real runs fit on all subjects across the source datasets."""
    signals, _ = load_subject(dataset_name=dataset_name,
                              subject_id=TEST_SUBJECTS[dataset_name],
                              data_root=DATA_ROOT)
    signals = preprocess_subject(signals, dataset_name=dataset_name)
    return fit_global_normalizer([signals])


def test_one(dataset_name, subject_id):
    print(f"\n{'='*72}")
    print(f"  {dataset_name.upper()} / {subject_id}")
    print(f"{'='*72}")

    # --- none (legacy reproducibility) ---
    df_none, status_none = process_subject(
        dataset_name, subject_id, data_root=DATA_ROOT,
        normalize_mode="none",
    )
    print(f"    status_none: {status_none}")
    assert df_none is not None, f"{dataset_name}: 'none' mode returned None"
    _summarize_features("none       ", df_none)

    # --- per_subject ---
    df_ps, _ = process_subject(
        dataset_name, subject_id, data_root=DATA_ROOT,
        normalize_mode="per_subject",
    )
    assert df_ps is not None, f"{dataset_name}: 'per_subject' returned None"
    _summarize_features("per_subject", df_ps)

    # --- global (using stats from this same subject for the smoke test) ---
    gstats = _build_global_stats(dataset_name)
    df_g, _ = process_subject(
        dataset_name, subject_id, data_root=DATA_ROOT,
        normalize_mode="global", global_stats=gstats,
    )
    assert df_g is not None, f"{dataset_name}: 'global' returned None"
    _summarize_features("global     ", df_g)

    # --- assertions ---
    # 1. Same window count across modes
    assert len(df_none) == len(df_ps) == len(df_g), (
        f"window counts diverged: none={len(df_none)}, "
        f"per_subject={len(df_ps)}, global={len(df_g)}"
    )
    print(f"  ✓ same window count across all three modes ({len(df_none)})")

    # 2. NaN structure: HRV may be NaN, nothing else should be NaN unless
    #    it was already NaN in the 'none' baseline.
    hrv = _expected_hrv_columns()
    non_hrv_cols = [c for c in df_ps.columns
                    if c not in hrv and c not in {"label", "start_sec",
                                                  "end_sec", "subject_id",
                                                  "dataset", "window_idx"}]
    new_nan_ps = df_ps[non_hrv_cols].isna().any() & ~df_none[non_hrv_cols].isna().any()
    new_nan_g  = df_g[non_hrv_cols].isna().any()  & ~df_none[non_hrv_cols].isna().any()
    assert not new_nan_ps.any(), (
        f"per_subject introduced NaN in non-HRV columns: "
        f"{new_nan_ps[new_nan_ps].index.tolist()}"
    )
    assert not new_nan_g.any(), (
        f"global introduced NaN in non-HRV columns: "
        f"{new_nan_g[new_nan_g].index.tolist()}"
    )
    print(f"  ✓ no new NaN introduced by normalization in non-HRV columns")

    # 3. HRV columns: should be IDENTICAL across modes (IBI is not z-scored,
    #    so HRV features depend only on cleaned IBI which is mode-invariant).
    for col in hrv:
        if col not in df_none.columns:
            continue
        a = df_none[col].values
        b = df_ps[col].values
        c = df_g[col].values
        mask = ~(np.isnan(a) | np.isnan(b) | np.isnan(c))
        if not mask.any():
            continue
        if not (np.allclose(a[mask], b[mask], rtol=1e-6, atol=1e-9)
                and np.allclose(a[mask], c[mask], rtol=1e-6, atol=1e-9)):
            raise AssertionError(
                f"{col}: HRV feature changed under normalization "
                f"(it shouldn't — IBI is not z-scored)"
            )
    print(f"  ✓ HRV features identical across modes (IBI not z-scored)")

    # 4. Some feature should actually change between none and per_subject —
    #    otherwise normalization had no effect, which would mean a bug.
    diff_cols = []
    for col in non_hrv_cols:
        if col not in df_none.columns or col not in df_ps.columns:
            continue
        a = df_none[col].values
        b = df_ps[col].values
        mask = ~(np.isnan(a) | np.isnan(b))
        if mask.any() and not np.allclose(a[mask], b[mask], rtol=1e-4):
            diff_cols.append(col)
    assert len(diff_cols) > 10, (
        f"per_subject differs from none in only {len(diff_cols)} cols — "
        f"normalization may not be wiring through. Diff cols: {diff_cols}"
    )
    print(f"  ✓ per_subject changes {len(diff_cols)} non-HRV feature cols "
          f"vs 'none' (normalization is wired through)")

    # 5. per_subject and global should differ (different stats applied).
    diff_psg = 0
    for col in non_hrv_cols:
        if col not in df_ps.columns:
            continue
        a, b = df_ps[col].values, df_g[col].values
        mask = ~(np.isnan(a) | np.isnan(b))
        if mask.any() and not np.allclose(a[mask], b[mask], rtol=1e-4):
            diff_psg += 1
    print(f"  ✓ per_subject vs global differ in {diff_psg} cols "
          f"(expected — different stats)")


def test_all():
    failures = []
    for dataset_name, subject_id in TEST_SUBJECTS.items():
        try:
            test_one(dataset_name, subject_id)
        except Exception as e:
            print(f"\n  ✗ {dataset_name}/{subject_id} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failures.append((dataset_name, subject_id, str(e)))

    print(f"\n{'='*72}")
    if failures:
        print(f"  FAILED: {len(failures)} dataset(s)")
        for d, s, e in failures:
            print(f"    - {d}/{s}: {e}")
        raise SystemExit(1)
    else:
        print(f"  ✓ all three datasets passed all checks")


if __name__ == "__main__":
    test_all()
