"""
Within-dataset baseline models: Random Forest, XGBoost, SVM (RBF).

Phase 5 of the cross-dataset stress detection project. Provides LOSO
(leave-one-subject-out) evaluation per dataset, with:
  - Median imputation for HRV NaN (fit on training fold, applied to test).
  - StandardScaler for SVM only (RF/XGB are scale-invariant).
  - Class weights to handle the imbalance that flips direction across datasets.
  - Optional decision-threshold tuning on a held-in validation slice
    (macro-F1 objective).
  - Small fixed hyperparameter grid per model, searched via inner GroupKFold.

Group convention for the nurse dataset: `subject_id` in features.csv is the
session_id (e.g. "83_1604630543"). LOSO must group by nurse, not session, or
sessions from the same nurse leak across folds. Callers pass a `groups` array
extracted as `session_id.split("_")[0]`; see scripts/run_within_dataset.py.

Imputation strategy:
    Eight HRV features have 32.9-38.7% NaN at the cohort level (genuine IBI
    sparsity, not a bug). We median-impute inside the sklearn Pipeline so
    the median is computed on the training fold only and applied to the
    held-out subject. The other 32 features are 0% NaN but the imputer is
    a no-op for them, so it's safe to apply uniformly.

Outputs:
    `loso_evaluate` returns a long-format DataFrame, one row per
    (model, held-out group, metric), suitable for groupby aggregation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# Metadata columns in features.csv — everything else is a feature.
# `nurse_id` is added by callers when grouping the Nurse dataset; include it
# here so it's never mistaken for a feature column.
META_COLUMNS = [
    "dataset", "subject_id", "window_idx", "start_sec", "end_sec", "label",
    "nurse_id",
]

# Number of inner folds for GridSearchCV-equivalent search.
INNER_CV_FOLDS = 3

# Held-in validation fraction used for threshold tuning.
THRESHOLD_TUNING_FRACTION = 0.15

# Threshold sweep resolution.
THRESHOLD_GRID = np.linspace(0.05, 0.95, 91)


# ----------------------------- model factories -----------------------------

def make_rf(class_weight="balanced", random_state=42, **overrides):
    """Random Forest pipeline. Imputer only (no scaler — RF is scale-invariant)."""
    params = dict(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
    )
    params.update(overrides)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(**params)),
    ])


def make_xgb(class_weight="balanced", random_state=42, **overrides):
    """
    XGBoost pipeline. Imputer only.

    XGBoost's `scale_pos_weight` is the imbalance lever (it doesn't accept
    sklearn's `class_weight=` string). We map "balanced" to the standard
    n_neg/n_pos ratio at fit time inside `_fit_xgb_with_weights`, not here,
    because the ratio depends on the training fold's composition.
    """
    if not HAS_XGB:
        raise ImportError(
            "xgboost not installed. Add `xgboost` to requirements.txt."
        )
    params = dict(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
    )
    params.update(overrides)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", XGBClassifier(**params)),
    ])


def make_svm(class_weight="balanced", random_state=42, **overrides):
    """SVM (RBF) pipeline. Imputer + StandardScaler — SVM needs scaling."""
    params = dict(
        C=1.0,
        gamma="scale",
        kernel="rbf",
        class_weight=class_weight,
        probability=True,
        random_state=random_state,
    )
    params.update(overrides)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", SVC(**params)),
    ])


# Hyperparameter grids. Keep small — these are baselines, not the final model.
# Reference: Schmidt 2018 (WESAD) and Campanella 2023 typical RF/SVM ranges.
HPARAM_GRIDS = {
    "rf": {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [None, 10],
        "clf__min_samples_leaf": [1, 4],
    },
    "xgb": {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [4, 6],
        "clf__learning_rate": [0.05, 0.1],
    },
    "svm": {
        "clf__C": [0.5, 1.0, 4.0],
        "clf__gamma": ["scale", 0.05],
    },
}

MODEL_FACTORIES = {
    "rf": make_rf,
    "svm": make_svm,
}
if HAS_XGB:
    MODEL_FACTORIES["xgb"] = make_xgb


# ----------------------------- helpers -----------------------------

def _split_features_and_meta(df):
    """Return (X, y, groups, subject_ids) from a features.csv subset."""
    feature_cols = [c for c in df.columns if c not in META_COLUMNS]
    X = df[feature_cols].values.astype(float)
    y = df["label"].values.astype(int)
    return X, y, feature_cols


def _xgb_scale_pos_weight(y_train):
    """Map class_weight='balanced' to XGBoost's scale_pos_weight = n_neg / n_pos."""
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    if n_pos == 0:
        return 1.0
    return n_neg / n_pos


def _set_xgb_imbalance(pipe, y_train):
    """Inject scale_pos_weight into an XGB pipeline before fitting."""
    if isinstance(pipe.named_steps["clf"], XGBClassifier if HAS_XGB else type(None)):
        pipe.named_steps["clf"].set_params(
            scale_pos_weight=_xgb_scale_pos_weight(y_train)
        )


def tune_threshold(y_true, y_proba, grid=THRESHOLD_GRID):
    """
    Pick the decision threshold that maximizes macro-F1 on a held-in slice.

    Returns the optimal threshold and the macro-F1 at it. Defaults to 0.5 if
    only one class is present (macro-F1 is degenerate).
    """
    if len(np.unique(y_true)) < 2:
        return 0.5, float("nan")

    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        y_pred = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t, best_f1


def _compute_metrics(y_true, y_pred, y_proba):
    """Compute metric dict for one fold.

    All metrics that average across classes pass `labels=[0, 1]` so the
    "absent class" gets F1=0 instead of being silently dropped from the
    average. Without this, a single-class fold where the model correctly
    predicts the present class scores macro_f1=1.0 (sklearn's default
    averages only over classes present in y_true). With it, macro_f1 caps
    at 0.5 for single-class folds — degenerate but honest.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true, y_pred, average="macro", labels=[0, 1], zero_division=0
        ),
    }

    # Per-class precision/recall/F1 — `labels=[0, 1]` so we get both rows
    # even if the held-out subject is single-class.
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    metrics.update({
        "precision_0": prec[0], "recall_0": rec[0], "f1_0": f1[0],
        "precision_1": prec[1], "recall_1": rec[1], "f1_1": f1[1],
    })

    # ROC-AUC requires both classes in y_true. Otherwise return NaN —
    # single-class held-out subjects are common in this dataset (Nurse
    # especially).
    if len(np.unique(y_true)) == 2 and y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")

    return metrics


# ----------------------------- fold-level eval -----------------------------

def evaluate_fold(
    model_name,
    X_train, y_train, groups_train,
    X_test, y_test,
    do_hparam_search=True,
    do_threshold_tuning=True,
    random_state=42,
):
    """
    Train and evaluate one LOSO fold.

    If `do_hparam_search`, runs inner GroupKFold over `groups_train` with
    the model's hparam grid, picks the macro-F1-best config, refits on all
    of (X_train, y_train).

    If `do_threshold_tuning`, carves a stratified slice out of the training
    fold (per THRESHOLD_TUNING_FRACTION), fits on the remainder, picks the
    macro-F1-optimal threshold on the held-in slice, then refits on the
    full training fold before predicting on test.

    Returns: (metrics_dict, best_threshold, best_params_or_None).
    """
    if model_name not in MODEL_FACTORIES:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(MODEL_FACTORIES)}"
        )

    factory = MODEL_FACTORIES[model_name]

    # --- 1. hyperparameter search on the training fold (inner GroupKFold) ---
    best_params = None
    if do_hparam_search:
        from sklearn.model_selection import GridSearchCV

        base = factory(random_state=random_state)
        if model_name == "xgb":
            _set_xgb_imbalance(base, y_train)

        # Inner CV groups must be a subset of the outer fold's training groups.
        n_groups = len(np.unique(groups_train))
        inner_n_splits = min(INNER_CV_FOLDS, n_groups)
        if inner_n_splits < 2:
            # Can't do CV with <2 groups — fall through to defaults.
            do_hparam_search = False
        else:
            cv = GroupKFold(n_splits=inner_n_splits)
            search = GridSearchCV(
                base,
                HPARAM_GRIDS[model_name],
                scoring="f1_macro",
                cv=cv.split(X_train, y_train, groups=groups_train),
                n_jobs=-1,
                refit=True,
                error_score="raise",
            )
            search.fit(X_train, y_train)
            pipe = search.best_estimator_
            best_params = search.best_params_

    if not do_hparam_search:
        pipe = factory(random_state=random_state)
        if model_name == "xgb":
            _set_xgb_imbalance(pipe, y_train)
        pipe.fit(X_train, y_train)

    # --- 2. threshold tuning on a held-in validation slice -----------------
    best_threshold = 0.5
    if do_threshold_tuning and len(np.unique(y_train)) == 2:
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=THRESHOLD_TUNING_FRACTION,
            random_state=random_state,
        )
        inner_idx, val_idx = next(sss.split(X_train, y_train))
        if len(np.unique(y_train[val_idx])) == 2:
            # Refit pipe on the 85% inner-train, predict on the 15% val.
            # We deliberately use the same hparams from the search above —
            # they were chosen on group-CV, so this slice is honest.
            pipe_tune = factory(random_state=random_state)
            if best_params is not None:
                pipe_tune.set_params(**best_params)
            if model_name == "xgb":
                _set_xgb_imbalance(pipe_tune, y_train[inner_idx])
            pipe_tune.fit(X_train[inner_idx], y_train[inner_idx])
            y_val_proba = pipe_tune.predict_proba(X_train[val_idx])[:, 1]
            best_threshold, _ = tune_threshold(y_train[val_idx], y_val_proba)

            # Refit final model on full training fold so we use all data.
            if model_name == "xgb":
                _set_xgb_imbalance(pipe, y_train)
            pipe.fit(X_train, y_train)

    # --- 3. predict on the held-out subject --------------------------------
    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= best_threshold).astype(int)

    metrics = _compute_metrics(y_test, y_pred, y_proba)
    return metrics, best_threshold, best_params, y_pred, y_proba


# ----------------------------- LOSO orchestrator ---------------------------

def loso_evaluate(
    features_df,
    model_name,
    group_col="subject_id",
    do_hparam_search=True,
    do_threshold_tuning=True,
    verbose=True,
    random_state=42,
):
    """
    Leave-one-group-out evaluation across a single dataset.

    `group_col` is the column to hold out one value of at a time. For
    Campanella and WESAD this is `subject_id`. For Nurse, the caller must
    add a `nurse_id` column (`session_id.split("_")[0]`) and pass
    `group_col="nurse_id"`.

    Returns a long-format DataFrame with columns:
        [model, held_out, n_test, n_train, best_threshold,
         <metric_columns...>]
    plus a `best_params` column (dict, may be None).

    The final row has `held_out == "POOLED"` and aggregates predictions
    across all valid folds: y_pred and y_proba are concatenated and one
    set of metrics is computed on the full pool. This matters for datasets
    like Nurse where most individual folds are single-class (so per-fold
    macro-F1 is degenerate) but the pool has both classes (so pooled
    macro-F1 is meaningful). See project_context.md Phase 5 notes.
    """
    X, y, feature_cols = _split_features_and_meta(features_df)
    groups = features_df[group_col].values

    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError(
            f"Need >=2 groups for LOSO; got {len(unique_groups)} "
            f"in column {group_col!r}."
        )

    rows = []
    # Collect across folds for pooled metric at the end.
    pooled_y_true, pooled_y_pred, pooled_y_proba = [], [], []

    for held_out in unique_groups:
        test_mask = groups == held_out
        train_mask = ~test_mask

        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        groups_train = groups[train_mask]

        # Skip folds where the training data is single-class (can't fit a
        # binary classifier). Record the skip so the summary reflects it.
        if len(np.unique(y_train)) < 2:
            if verbose:
                print(f"  [skip] {held_out}: training data is single-class")
            continue

        try:
            metrics, thr, best_params, y_pred, y_proba = evaluate_fold(
                model_name,
                X_train, y_train, groups_train,
                X_test, y_test,
                do_hparam_search=do_hparam_search,
                do_threshold_tuning=do_threshold_tuning,
                random_state=random_state,
            )
        except Exception as e:
            if verbose:
                print(f"  [error] {held_out}: {e}")
            continue

        # Accumulate for pooled metric.
        pooled_y_true.append(y_test)
        pooled_y_pred.append(y_pred)
        pooled_y_proba.append(y_proba)

        row = {
            "model": model_name,
            "held_out": str(held_out),
            "n_test": int(test_mask.sum()),
            "n_train": int(train_mask.sum()),
            "n_test_pos": int(np.sum(y_test == 1)),
            "n_test_neg": int(np.sum(y_test == 0)),
            "best_threshold": thr,
            "best_params": best_params,
            **metrics,
        }
        rows.append(row)

        if verbose:
            print(
                f"  {held_out:>20s}  "
                f"acc={metrics['accuracy']:.3f}  "
                f"macro_f1={metrics['macro_f1']:.3f}  "
                f"bal_acc={metrics['balanced_accuracy']:.3f}  "
                f"thr={thr:.2f}"
            )

    # --- pooled metric row --------------------------------------------------
    if pooled_y_true:
        yt = np.concatenate(pooled_y_true)
        yp = np.concatenate(pooled_y_pred)
        ypr = np.concatenate(pooled_y_proba)
        pooled_metrics = _compute_metrics(yt, yp, ypr)
        pooled_row = {
            "model": model_name,
            "held_out": "POOLED",
            "n_test": int(len(yt)),
            "n_train": float("nan"),  # not meaningful for pooled; promotes column to float
            "n_test_pos": int(np.sum(yt == 1)),
            "n_test_neg": int(np.sum(yt == 0)),
            "best_threshold": float("nan"),  # mixture of per-fold thresholds
            "best_params": None,
            **pooled_metrics,
        }
        rows.append(pooled_row)
        if verbose:
            print(
                f"  {'POOLED':>20s}  "
                f"acc={pooled_metrics['accuracy']:.3f}  "
                f"macro_f1={pooled_metrics['macro_f1']:.3f}  "
                f"bal_acc={pooled_metrics['balanced_accuracy']:.3f}  "
                f"(n={len(yt)}, pos={int(np.sum(yt == 1))}, neg={int(np.sum(yt == 0))})"
            )

    return pd.DataFrame(rows)


def summarize_loso(results_df, by=("model",)):
    """
    Aggregate per-fold results: mean ± std across held-out groups,
    grouped by `by` (e.g. ("model",) or ("dataset", "model")).

    Excludes the POOLED row from per-fold aggregation (it's already an
    aggregate; including it would double-count and skew the mean/std).
    """
    metric_cols = [
        "accuracy", "balanced_accuracy", "macro_f1",
        "precision_0", "recall_0", "f1_0",
        "precision_1", "recall_1", "f1_1",
        "roc_auc",
    ]
    metric_cols = [c for c in metric_cols if c in results_df.columns]

    per_fold = results_df[results_df["held_out"] != "POOLED"]
    agg = per_fold.groupby(list(by))[metric_cols].agg(["mean", "std", "count"])
    return agg
