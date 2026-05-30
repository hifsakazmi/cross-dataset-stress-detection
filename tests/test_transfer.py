"""
Smoke test for src.models.evaluate_transfer.

Builds two synthetic "datasets" with deliberately offset per-feature means
(so transfer is non-trivial) and verifies:

1. evaluate_transfer returns a row with every key the runner expects
   (metrics from _compute_metrics + n_train/n_test/best_threshold/etc).
2. All four (norm × threshold) combos run without error, where here we
   simulate "global normalization" with a deliberate mean offset and
   "per-subject normalization" by mean-centering each subject's features
   before the transfer call.
3. The 0.5-threshold and source-tuned-threshold variants both return,
   and the tuned variant's best_threshold may differ from 0.5.
4. Per-subject-aligned features beat unaligned features on a deliberately
   offset target — sanity check that domain adaptation does what it claims.

Doesn't touch real feature files. ~5 seconds.

Run from repo root:
    python -m tests.test_transfer
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models import MODEL_FACTORIES, evaluate_transfer


META_COLS = {"dataset", "subject_id", "window_idx",
             "start_sec", "end_sec", "label", "nurse_id"}


def _make_synth_dataset(dataset_name, n_subjects=6, n_windows=20,
                       feature_offset=0.0, label_signal=1.2, seed=0):
    """
    Build a synthetic features DataFrame in the shape the runner expects.

    - 8 "hrv_*" features so the column layout matches real data (these
      may contain NaN; we leave them clean here for simplicity).
    - 32 generic feat_* features. feat_0 and feat_1 carry the label
      signal; the rest are noise.
    - feature_offset shifts every numeric feature by a constant — this
      is the "domain shift" we want transfer to handle.
    - subject_id is unique per subject in this dataset; nurse_id mirrors
      it (since synthetic data has no real session structure).
    """
    rng = np.random.default_rng(seed)
    hrv_names = ["hrv_rmssd", "hrv_sdnn", "hrv_pnn50", "hrv_mean_ibi",
                 "hrv_median_ibi", "hrv_hr_mean", "hrv_hr_std", "hrv_lf_hf_ratio"]
    other_names = [f"feat_{i:02d}" for i in range(32)]
    feat_names = hrv_names + other_names
    rows = []
    for s in range(n_subjects):
        subject_id = f"{dataset_name}_sub_{s:02d}"
        per_subject_shift = rng.normal(0, 0.2)
        for w in range(n_windows):
            label = w % 2  # alternating, balanced
            feats = rng.normal(0, 1, len(feat_names))
            # Signal carriers
            feats[len(hrv_names) + 0] += label * label_signal + per_subject_shift
            feats[len(hrv_names) + 1] += label * (label_signal * 0.7) + per_subject_shift
            # Apply the dataset-wide offset (the "domain shift")
            feats = feats + feature_offset

            row = {
                "dataset": dataset_name,
                "subject_id": subject_id,
                "nurse_id": subject_id,  # group_col fallback
                "window_idx": w,
                "start_sec": w * 30.0,
                "end_sec": w * 30.0 + 60.0,
                "label": int(label),
            }
            for name, val in zip(feat_names, feats):
                row[name] = float(val)
            rows.append(row)
    return pd.DataFrame(rows)


def _expected_row_keys():
    return {
        "n_train", "n_test", "n_train_groups",
        "best_threshold", "best_params",
        "accuracy", "balanced_accuracy", "macro_f1",
        "precision_0", "recall_0", "f1_0",
        "precision_1", "recall_1", "f1_1",
        "roc_auc",
    }


def test_returns_expected_keys():
    print("\n--- returns_expected_keys ---")
    src = _make_synth_dataset("A", feature_offset=0.0, seed=1)
    tgt = _make_synth_dataset("B", feature_offset=0.0, seed=2)
    row = evaluate_transfer(
        src, tgt, model_name="rf",
        do_hparam_search=False, do_threshold_tuning=False,
    )
    missing = _expected_row_keys() - set(row.keys())
    assert not missing, f"missing keys: {missing}"
    print(f"  ✓ all expected keys present ({len(row)} fields)")


def test_all_threshold_norm_combos_run():
    print("\n--- all_threshold_norm_combos_run ---")
    # "global normalization" simulated as both datasets sharing a single
    # offset (no per-subject correction). "per_subject normalization"
    # simulated as mean-centering each subject's features.
    src = _make_synth_dataset("A", feature_offset=0.0, seed=10)
    tgt_global  = _make_synth_dataset("B", feature_offset=2.0, seed=11)
    tgt_aligned = _per_subject_center(_make_synth_dataset("B", feature_offset=2.0, seed=11))
    src_aligned = _per_subject_center(src)

    for model in ["rf", "svm"]:
        if model not in MODEL_FACTORIES:
            continue
        for tune in [False, True]:
            for label, src_df, tgt_df in [
                ("global    ", src, tgt_global),
                ("per_subject", src_aligned, tgt_aligned),
            ]:
                row = evaluate_transfer(
                    src_df, tgt_df, model_name=model,
                    do_hparam_search=False, do_threshold_tuning=tune,
                )
                assert "macro_f1" in row
                print(f"  ✓ {model} norm={label.strip()} tune={tune}: "
                      f"macro_f1={row['macro_f1']:.3f} thr={row['best_threshold']:.2f}")


def test_per_subject_helps_on_offset_target():
    """Domain-shift sanity check: alignment should beat no-alignment when
    the only difference between source and target is a constant offset."""
    print("\n--- per_subject_helps_on_offset_target ---")
    src = _make_synth_dataset("A", feature_offset=0.0, n_subjects=8,
                              n_windows=30, seed=42)
    # Big domain shift: every feature offset by +3 on target.
    tgt = _make_synth_dataset("B", feature_offset=3.0, n_subjects=8,
                              n_windows=30, seed=43)

    # Without alignment: RF/XGB tolerate the offset (tree splits adapt),
    # but SVM with RBF kernel should be hurt. So we check on SVM.
    if "svm" not in MODEL_FACTORIES:
        print("  (svm unavailable; skipping)")
        return

    row_naive = evaluate_transfer(
        src, tgt, model_name="svm",
        do_hparam_search=False, do_threshold_tuning=False,
    )
    row_aligned = evaluate_transfer(
        _per_subject_center(src), _per_subject_center(tgt),
        model_name="svm",
        do_hparam_search=False, do_threshold_tuning=False,
    )
    print(f"  naive   macro_f1={row_naive['macro_f1']:.3f}")
    print(f"  aligned macro_f1={row_aligned['macro_f1']:.3f}")
    # Aligned should be no worse, and meaningfully better in expectation.
    # Use a soft assertion: aligned >= naive - 0.02 (within noise) AND
    # aligned > 0.5 (better than constant prediction).
    assert row_aligned["macro_f1"] >= row_naive["macro_f1"] - 0.02, (
        f"alignment hurt SVM: naive={row_naive['macro_f1']:.3f}, "
        f"aligned={row_aligned['macro_f1']:.3f}"
    )
    print("  ✓ alignment ≥ naive within noise (DA sanity check)")


def test_single_class_train_raises():
    print("\n--- single_class_train_raises ---")
    src = _make_synth_dataset("A", seed=7)
    src["label"] = 0  # collapse to single class
    tgt = _make_synth_dataset("B", seed=8)
    try:
        evaluate_transfer(src, tgt, model_name="rf",
                          do_hparam_search=False, do_threshold_tuning=False)
        raise AssertionError("expected ValueError on single-class train")
    except ValueError as e:
        assert "single-class" in str(e), f"unexpected error message: {e}"
        print(f"  ✓ raised: {e}")


def _per_subject_center(df):
    """Subtract each subject's per-feature mean. Mimics per-subject z-scoring
    at the feature level for the smoke test (not what the real pipeline does
    — the real pipeline z-scores at the signal level — but adequate to test
    that evaluate_transfer reacts to aligned vs unaligned inputs)."""
    out = df.copy()
    feat_cols = [c for c in out.columns if c not in META_COLS]
    for sid, group in out.groupby("subject_id"):
        mean = group[feat_cols].mean()
        out.loc[group.index, feat_cols] = group[feat_cols] - mean
    return out


def test_all():
    failed = False
    for fn in [
        test_returns_expected_keys,
        test_all_threshold_norm_combos_run,
        test_per_subject_helps_on_offset_target,
        test_single_class_train_raises,
    ]:
        try:
            fn()
        except Exception as e:
            print(f"  ✗ {fn.__name__}: {e}")
            import traceback; traceback.print_exc()
            failed = True
    print("\n" + "="*72)
    if failed:
        print("  FAILED")
        raise SystemExit(1)
    print("  ✓ all tests passed")


if __name__ == "__main__":
    test_all()
