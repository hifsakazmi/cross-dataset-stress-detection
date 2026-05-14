"""
Smoke test for src.models.

Builds a synthetic features DataFrame (mimicking the real features.csv schema)
with 4 fake subjects, 60 windows each, 40 features. Runs LOSO for RF and SVM
on it; XGB if installed. Confirms:

  - Output DataFrame has the expected shape and columns.
  - All folds produce metrics in valid ranges.
  - Imputation handles NaN HRV features (we inject some).
  - Threshold tuning runs without error.
  - Single-class folds are skipped, not silently scored 0.

This does NOT validate real-data results — that's the runner script's job.
The point here is to catch type/shape regressions in src.models.

Run from repo root:
    python -m tests.test_models
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models import (
    HAS_XGB,
    MODEL_FACTORIES,
    loso_evaluate,
    summarize_loso,
    tune_threshold,
)


HRV_FEATURES = [
    "hrv_rmssd", "hrv_sdnn", "hrv_pnn50",
    "hrv_mean_ibi", "hrv_median_ibi",
    "hrv_hr_mean", "hrv_hr_std", "hrv_lf_hf_ratio",
]


def make_synthetic_features(n_subjects=4, n_windows_per_subject=60,
                            n_features=40, hrv_nan_rate=0.3, seed=42):
    """
    Build a fake features DataFrame matching features.csv schema.

    The label is correlated with a couple of features so RF/SVM actually
    have signal to learn — otherwise threshold tuning would be meaningless.
    """
    rng = np.random.default_rng(seed)

    rows = []
    feat_names = HRV_FEATURES + [f"feat_{i:02d}" for i in range(n_features - 8)]

    for s in range(n_subjects):
        for w in range(n_windows_per_subject):
            # Half stress, half non-stress per subject. Add per-subject offset
            # so LOSO has something subject-specific to generalize across.
            label = w % 2
            subject_offset = rng.normal(0, 0.3)

            features = rng.normal(0, 1, n_features)
            # Inject signal: features[0] and features[1] shift with label.
            features[0] += label * 1.2 + subject_offset
            features[1] += label * 0.8 + subject_offset

            # Sprinkle NaN into HRV features
            for i in range(8):
                if rng.random() < hrv_nan_rate:
                    features[i] = np.nan

            row = {
                "dataset": "synthetic",
                "subject_id": f"sub_{s:02d}",
                "window_idx": w,
                "start_sec": w * 30.0,
                "end_sec": w * 30.0 + 60.0,
                "label": label,
            }
            for name, val in zip(feat_names, features):
                row[name] = val
            rows.append(row)

    return pd.DataFrame(rows)


def test_tune_threshold():
    print("\n--- tune_threshold ---")
    rng = np.random.default_rng(0)
    # Synthetic: probabilities well-calibrated, true labels match >0.5
    y_true = (rng.random(200) > 0.4).astype(int)
    y_proba = np.clip(y_true + rng.normal(0, 0.2, 200), 0, 1)
    thr, f1 = tune_threshold(y_true, y_proba)
    assert 0.05 <= thr <= 0.95, f"threshold out of bounds: {thr}"
    assert 0.0 <= f1 <= 1.0, f"f1 out of bounds: {f1}"
    print(f"  picked threshold {thr:.2f}, macro-F1 {f1:.3f}  ✓")

    # Single-class case — should fall back to 0.5
    y_single = np.zeros(100, dtype=int)
    y_proba_single = rng.random(100)
    thr, f1 = tune_threshold(y_single, y_proba_single)
    assert thr == 0.5
    print(f"  single-class fallback to 0.5  ✓")


def test_loso_smoke():
    print("\n--- loso_evaluate (smoke) ---")
    df = make_synthetic_features(n_subjects=4, n_windows_per_subject=60)
    print(f"  Synthetic features: {df.shape}")
    print(f"  HRV NaN rate: "
          f"{df[HRV_FEATURES].isna().mean().mean():.1%} (target ~30%)")

    expected_metrics = [
        "accuracy", "balanced_accuracy", "macro_f1",
        "precision_0", "recall_0", "f1_0",
        "precision_1", "recall_1", "f1_1",
        "roc_auc",
    ]

    for model_name in MODEL_FACTORIES:
        print(f"\n  Model: {model_name}")
        results = loso_evaluate(
            df,
            model_name=model_name,
            group_col="subject_id",
            do_hparam_search=False,    # keep smoke test fast
            do_threshold_tuning=True,
            verbose=False,
        )

        # Output shape: 4 per-fold rows + 1 POOLED row = 5 total
        assert len(results) == 5, f"expected 5 rows (4 folds + POOLED), got {len(results)}"
        assert "POOLED" in results["held_out"].values, "missing POOLED row"
        per_fold = results[results["held_out"] != "POOLED"]
        pooled = results[results["held_out"] == "POOLED"]
        assert len(per_fold) == 4
        assert len(pooled) == 1
        print(f"  → {len(per_fold)} folds + 1 POOLED row")

        # Required columns
        for col in expected_metrics + ["model", "held_out", "best_threshold"]:
            assert col in results.columns, f"missing column: {col}"

        # Metrics in valid ranges (per-fold rows only — pooled has same range)
        for col in ["accuracy", "balanced_accuracy", "macro_f1"]:
            vals = results[col].values
            assert ((vals >= 0) & (vals <= 1)).all(), \
                f"{col} out of [0,1]: {vals}"

        # Threshold in sweep range (per-fold only — POOLED has NaN by design)
        thrs = per_fold["best_threshold"].values
        assert ((thrs >= 0.05) & (thrs <= 0.95)).all(), \
            f"threshold out of range: {thrs}"

        # Model has *some* skill on this easy synthetic task
        mean_f1 = per_fold["macro_f1"].mean()
        assert mean_f1 > 0.5, \
            f"{model_name} should beat 0.5 macro-F1 on synthetic; got {mean_f1:.3f}"

        # Pooled macro-F1 should also be above 0.5 (synthetic data has both classes)
        pooled_f1 = pooled["macro_f1"].iloc[0]
        assert pooled_f1 > 0.5, \
            f"{model_name} pooled macro-F1 should beat 0.5; got {pooled_f1:.3f}"

        print(f"  ✓ per-fold mean macro-F1 = {mean_f1:.3f}, "
              f"pooled macro-F1 = {pooled_f1:.3f}, "
              f"thresholds = [{thrs.min():.2f}, {thrs.max():.2f}]")


def test_single_class_macro_f1_capped():
    """A single-class held-out fold's macro-F1 must cap at 0.5, not inflate to 1.0.

    Regression test for the bug we hit on Nurse fold BG: when y_true is uniformly
    class 1 and the model predicts class 1 for everything, sklearn's default
    average='macro' silently dropped the absent class and returned 1.0. With
    labels=[0, 1] passed explicitly, macro_f1 = (F1_0 + F1_1) / 2 = (0 + 1) / 2 = 0.5.
    """
    print("\n--- single-class macro-F1 is capped at 0.5 (regression) ---")
    from src.models import _compute_metrics
    yt = np.ones(50, dtype=int)
    yp = np.ones(50, dtype=int)
    ypr = np.ones(50, dtype=float)
    m = _compute_metrics(yt, yp, ypr)
    assert m["macro_f1"] == 0.5, \
        f"single-class macro_f1 should be 0.5, got {m['macro_f1']}"
    assert m["accuracy"] == 1.0, f"accuracy should be 1.0, got {m['accuracy']}"
    print(f"  ✓ all-1s y_true, all-1s y_pred: macro_f1=0.5 (not 1.0), acc=1.0")


def test_summarize():
    print("\n--- summarize_loso ---")
    df = make_synthetic_features(n_subjects=3, n_windows_per_subject=40)
    results = loso_evaluate(
        df, model_name="rf", group_col="subject_id",
        do_hparam_search=False, do_threshold_tuning=False, verbose=False,
    )
    # Results contain 3 per-fold rows + 1 POOLED row = 4 total
    assert len(results) == 4
    summary = summarize_loso(results, by=("model",))
    assert ("macro_f1", "mean") in summary.columns
    assert ("macro_f1", "std") in summary.columns
    # summarize_loso must exclude POOLED — count should equal 3 (per-fold), not 4
    assert summary[("macro_f1", "count")].iloc[0] == 3, \
        f"expected count=3 (POOLED excluded), got {summary[('macro_f1', 'count')].iloc[0]}"
    print(f"  ✓ summary excludes POOLED (count={int(summary[('macro_f1', 'count')].iloc[0])}); "
          f"macro_f1 mean = {summary[('macro_f1', 'mean')].iloc[0]:.3f}")


def test_single_class_fold_skipped():
    """If a fold's training data is single-class, it should be skipped."""
    print("\n--- single-class fold handling ---")
    df = make_synthetic_features(n_subjects=3, n_windows_per_subject=20)

    # Force subject sub_00's TRAINING data (i.e. sub_01 + sub_02) to be
    # single-class by overwriting labels for those subjects.
    mask = df["subject_id"].isin(["sub_01", "sub_02"])
    df.loc[mask, "label"] = 1

    results = loso_evaluate(
        df, model_name="rf", group_col="subject_id",
        do_hparam_search=False, do_threshold_tuning=False, verbose=False,
    )
    # sub_00 is held out; training = sub_01 + sub_02 = all label=1, single-class.
    # That fold should be skipped. Other two folds (sub_01, sub_02 held out)
    # have mixed labels in their training and should run.
    held_out = set(results["held_out"])
    assert "sub_00" not in held_out, \
        f"single-class training fold should be skipped, got {held_out}"
    print(f"  ✓ single-class training fold dropped; kept folds: {sorted(held_out)}")


if __name__ == "__main__":
    print("=" * 72)
    print("  src.models smoke test")
    print(f"  XGBoost available: {HAS_XGB}")
    print(f"  MODEL_FACTORIES: {list(MODEL_FACTORIES.keys())}")
    print("=" * 72)

    test_tune_threshold()
    test_loso_smoke()
    test_single_class_macro_f1_capped()
    test_summarize()
    test_single_class_fold_skipped()

    print("\n" + "=" * 72)
    print("  ALL TESTS PASSED")
    print("=" * 72)
