"""
Smoke test for src.preprocessing.

Loads one subject per dataset, runs preprocess_subject, and prints
shape + min/max/mean for each signal before and after cleaning.
Also tests normalize_signals on the cleaned output.

Run from repo root:
    python -m tests.test_preprocessing
"""

import numpy as np

from src.data_loader import load_subject
from src.preprocessing import preprocess_subject, normalize_signals


# One representative subject per dataset. Match the choices used in
# notebooks/01_eda.ipynb so output is comparable.
TEST_SUBJECTS = {
    "campanella": "subject_01",
    "wesad": "S2",
    "nurse": "83_1604630543",
}

DATA_ROOT = "data_extracted"


def _stats_line(name, df):
    """One-line summary of a signal DataFrame."""
    if df is None or len(df) == 0:
        return f"    {name:5s}  EMPTY"
    # IBI has columns [offset, ibi]; summarize only the ibi column,
    # otherwise the giant offsets pollute the stats.
    if name == "IBI":
        col = "ibi" if "ibi" in df.columns else df.columns[-1]
        values = df[[col]].values.astype(float)
        shape_str = f"({len(df)},)"
    else:
        values = df.values.astype(float)
        shape_str = str(df.shape)
    return (
        f"    {name:5s}  shape={shape_str:12s}  "
        f"min={np.nanmin(values):8.3f}  "
        f"max={np.nanmax(values):8.3f}  "
        f"mean={np.nanmean(values):8.3f}  "
        f"std={np.nanstd(values):8.3f}"
    )


def _print_signals(label, signals):
    print(f"  {label}")
    for name in ["ACC", "BVP", "EDA", "HR", "IBI", "TEMP"]:
        if name in signals:
            print(_stats_line(name, signals[name]))


def _check_invariants(raw, cleaned, dataset_name):
    """Assert no silent breakage."""
    issues = []

    for name, raw_df in raw.items():
        if name not in cleaned:
            issues.append(f"{name}: dropped during preprocessing")
            continue
        clean_df = cleaned[name]

        if clean_df is None or len(clean_df) == 0:
            if len(raw_df) > 0:
                issues.append(f"{name}: became empty after preprocessing")
            continue

        # Length should match raw except for Campanella BVP (11-sample trim)
        # and IBI (physiological rejection can drop rows).
        expected_len = len(raw_df)
        if name == "BVP" and dataset_name == "campanella":
            expected_len = len(raw_df) - 11
        if name == "IBI":
            # IBI may drop intervals outside 0.3-2.0 sec; just check it didn't
            # lose everything
            if len(clean_df) == 0 and len(raw_df) > 0:
                issues.append(f"IBI: all intervals rejected")
            continue

        if len(clean_df) != expected_len:
            issues.append(
                f"{name}: length changed unexpectedly "
                f"({len(raw_df)} -> {len(clean_df)}, expected {expected_len})"
            )

        # No all-NaN output
        if np.all(np.isnan(clean_df.values)):
            issues.append(f"{name}: all NaN after preprocessing")

        # TEMP must be within clip range
        if name == "TEMP":
            vals = clean_df.values
            if np.nanmax(vals) > 40.0 + 1e-6 or np.nanmin(vals) < 20.0 - 1e-6:
                issues.append(
                    f"TEMP: clip failed (min={np.nanmin(vals)}, max={np.nanmax(vals)})"
                )

    if issues:
        print(f"  ⚠ ISSUES:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print(f"  ✓ all invariants pass")


def _check_normalization(normalized):
    """After per-subject z-score, non-IBI signals should be ~mean 0, std 1."""
    print(f"  Normalization check (per-subject z-score):")
    for name in ["ACC", "BVP", "EDA", "HR", "TEMP"]:
        if name not in normalized or len(normalized[name]) == 0:
            continue
        vals = normalized[name].values.astype(float)
        m = np.nanmean(vals)
        s = np.nanstd(vals)
        ok = abs(m) < 1e-6 and abs(s - 1.0) < 1e-3
        flag = "✓" if ok else "✗"
        print(f"    {flag} {name:5s}  mean={m:+.6f}  std={s:.6f}")


def test_preprocessing_one_subject(dataset_name, subject_id):
    print(f"\n{'='*70}")
    print(f"  {dataset_name.upper()} / {subject_id}")
    print(f"{'='*70}")

    signals_raw, sampling_rates = load_subject(
        dataset_name=dataset_name,
        subject_id=subject_id,
        data_root=DATA_ROOT,
    )

    if not signals_raw:
        print(f"  ⚠ No signals loaded for {subject_id}")
        return

    _print_signals("BEFORE", signals_raw)

    signals_clean = preprocess_subject(signals_raw, dataset_name=dataset_name)

    _print_signals("AFTER", signals_clean)

    _check_invariants(signals_raw, signals_clean, dataset_name)

    signals_norm = normalize_signals(signals_clean, mode="per_subject")
    _check_normalization(signals_norm)


def test_all():
    for dataset_name, subject_id in TEST_SUBJECTS.items():
        try:
            test_preprocessing_one_subject(dataset_name, subject_id)
        except Exception as e:
            print(f"\n  ✗ {dataset_name}/{subject_id} failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    test_all()
