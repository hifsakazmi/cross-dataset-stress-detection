"""
Preprocessing module: per-signal cleaning and normalization.

Operates on the DataFrame outputs of data_loader.load_subject, BEFORE windowing.
Preserving the DatetimeIndex matters — Campanella's BVP trim re-anchors the
index forward by 11/64 s so downstream windowing slices stay correct, and
filter operations need uniform sampling which the index documents.

Pipeline order:
    load_subject() -> preprocess_subject() -> create_windows() -> extract_features()

Normalization is intentionally NOT applied inside preprocess_subject — it must
be fit on training subjects and applied to test subjects in the CV loop, so
normalize_signals() is exposed separately.
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

# E4 sampling rates — duplicated from data_loader rather than imported to keep
# preprocessing self-contained for testing. Must stay in sync.
E4_SAMPLING_RATES = {
    "ACC": 32,
    "BVP": 64,
    "EDA": 4,
    "HR": 1,
    "TEMP": 4,
}

# Physiological clip ranges
TEMP_MIN_C = 20.0
TEMP_MAX_C = 40.0

# IBI rejection range (300-2000 ms = 30-200 bpm)
IBI_MIN_SEC = 0.3
IBI_MAX_SEC = 2.0

# Campanella BVP warmup: first 11 samples are zeros across all 29 subjects
CAMPANELLA_BVP_TRIM_SAMPLES = 11


# ---------- Per-signal cleaners ----------

def _butter_filter(data, cutoff, fs, btype, order=4):
    """Zero-phase Butterworth filter via filtfilt."""
    nyq = fs / 2
    if isinstance(cutoff, (list, tuple)):
        wn = [c / nyq for c in cutoff]
    else:
        wn = cutoff / nyq
    b, a = butter(order, wn, btype=btype)
    return filtfilt(b, a, data, axis=0)


def clean_bvp(df, fs=64):
    """Bandpass 0.5-8 Hz, 4th-order Butterworth, zero-phase."""
    if len(df) < 30:  # filtfilt needs ~3x filter order samples; be safe
        return df
    values = df.values.astype(float).squeeze()
    filtered = _butter_filter(values, [0.5, 8.0], fs, btype="bandpass", order=4)
    return pd.DataFrame(filtered, index=df.index, columns=df.columns)


def clean_eda(df, fs=4):
    """Lowpass 1 Hz to suppress motion noise. SCR components live below 1 Hz."""
    if len(df) < 30:
        return df
    values = df.values.astype(float).squeeze()
    # 1 Hz cutoff at fs=4 means wn=0.5 — right at Nyquist, butter is fine
    filtered = _butter_filter(values, 1.0, fs, btype="lowpass", order=4)
    return pd.DataFrame(filtered, index=df.index, columns=df.columns)


def clean_acc(df, fs=32):
    """Lowpass 10 Hz to suppress high-freq noise. Per-axis."""
    if len(df) < 30:
        return df
    values = df.values.astype(float)
    filtered = _butter_filter(values, 10.0, fs, btype="lowpass", order=4)
    return pd.DataFrame(filtered, index=df.index, columns=df.columns)


def clean_temp(df):
    """Clip to physiological range (handles WESAD warmup 382C artifacts)."""
    clipped = df.clip(lower=TEMP_MIN_C, upper=TEMP_MAX_C)
    return clipped


def clean_hr(df):
    """No filtering — E4 firmware already smooths HR. Pass-through."""
    return df


def clean_ibi(df):
    """Reject physiologically implausible intervals (HR < 30 or > 200 bpm)."""
    if len(df) == 0:
        return df
    # IBI DataFrame has columns ["offset", "ibi"] per data_loader.read_e4_ibi
    ibi_col = "ibi" if "ibi" in df.columns else df.columns[-1]
    mask = (df[ibi_col] >= IBI_MIN_SEC) & (df[ibi_col] <= IBI_MAX_SEC)
    return df[mask].copy()


# ---------- Dataset-specific quirks ----------

def trim_campanella_bvp(df, fs=64, n_samples=CAMPANELLA_BVP_TRIM_SAMPLES):
    """
    Trim the first n_samples of BVP (warmup zeros) and DO NOT shift the
    timestamp index. Dropping rows from the head naturally leaves the
    remaining timestamps unchanged, so windowing still slices by the same
    absolute times. The recording effectively starts ~n_samples/fs seconds
    later for BVP than for other signals.
    """
    if len(df) <= n_samples:
        return df  # don't trim past the end; defensive
    return df.iloc[n_samples:].copy()


# ---------- Top-level dispatch ----------

def preprocess_subject(signals, dataset_name):
    """
    Apply per-signal cleaning + dataset-specific quirks. Returns a new dict;
    does not mutate the input.

    Args:
        signals:       dict of signal_name -> DataFrame, as returned by load_subject
        dataset_name:  "campanella" | "wesad" | "nurse"

    Returns:
        dict of signal_name -> cleaned DataFrame (same keys as input)
    """
    out = {}

    for name, df in signals.items():
        if df is None or len(df) == 0:
            out[name] = df
            continue

        cleaned = df

        # Dataset-specific BVP trim BEFORE filtering (so the filter doesn't ring
        # off the zero-to-real-signal step)
        if name == "BVP" and dataset_name == "campanella":
            cleaned = trim_campanella_bvp(cleaned, fs=E4_SAMPLING_RATES["BVP"])

        # Per-signal cleaning
        if name == "BVP":
            cleaned = clean_bvp(cleaned, fs=E4_SAMPLING_RATES["BVP"])
        elif name == "EDA":
            cleaned = clean_eda(cleaned, fs=E4_SAMPLING_RATES["EDA"])
        elif name == "ACC":
            cleaned = clean_acc(cleaned, fs=E4_SAMPLING_RATES["ACC"])
        elif name == "TEMP":
            cleaned = clean_temp(cleaned)
        elif name == "HR":
            cleaned = clean_hr(cleaned)
        elif name == "IBI":
            cleaned = clean_ibi(cleaned)

        out[name] = cleaned

    return out


# ---------- Normalization ----------

def fit_normalizer(signals, mode="per_subject"):
    """
    Compute mean/std per signal for later application. For per_subject mode,
    this is called once per subject and the result is used to transform that
    subject's signals only. For global mode, statistics should be aggregated
    across all training subjects (caller's responsibility).

    Returns:
        dict of signal_name -> (mean, std) tuples. std is clipped to >= 1e-8
        to prevent division by zero on flat signals.
    """
    stats = {}
    for name, df in signals.items():
        if df is None or len(df) == 0 or name == "IBI":
            # IBI is event-based, not z-scored. HRV features handle their own
            # normalization implicitly (RMSSD, pNN50 are already scale-aware).
            continue
        values = df.values.astype(float)
        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        stats[name] = (mean, std)
    return stats


def apply_normalizer(signals, stats):
    """Apply z-score transform using pre-fit stats. Returns new dict."""
    out = {}
    for name, df in signals.items():
        if df is None or len(df) == 0 or name not in stats:
            out[name] = df
            continue
        mean, std = stats[name]
        values = (df.values.astype(float) - mean) / std
        out[name] = pd.DataFrame(values, index=df.index, columns=df.columns)
    return out


def normalize_signals(signals, mode="per_subject", global_stats=None):
    """
    Convenience wrapper.

    Args:
        signals:       cleaned signals dict
        mode:          "per_subject" (default) or "global"
        global_stats:  required if mode="global" — dict from fit_normalizer
                       called on the pooled training set

    Returns:
        normalized signals dict
    """
    if mode == "per_subject":
        stats = fit_normalizer(signals)
        return apply_normalizer(signals, stats)
    elif mode == "global":
        if global_stats is None:
            raise ValueError("global mode requires precomputed global_stats")
        return apply_normalizer(signals, global_stats)
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")