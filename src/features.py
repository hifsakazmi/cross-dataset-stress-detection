"""
Feature extraction: turns windowed signals into a flat feature DataFrame.

Operates on the output of create_windows (a list of dicts with numpy arrays
under "signals"). Sampling rates are NOT embedded in the windows, so they're
passed in alongside.

Feature inventory (~40 features per window):
    HRV from IBI (8):  RMSSD, SDNN, pNN50, mean IBI, median IBI,
                       HR mean, HR std, LF/HF ratio
    EDA (10):          mean, std, min, max, slope, SCR peak count,
                       mean SCR amplitude, mean SCR rise time,
                       tonic mean (SCL), phasic std
    BVP (6):           mean, std, spectral entropy, dominant freq,
                       power in 0.04-0.15 Hz, power in 0.15-0.4 Hz
    HR (4):            mean, std, min, max
    ACC (8):           magnitude mean/std/min/max, magnitude AUC,
                       jerk std, dominant freq, signal magnitude area (SMA)
    TEMP (4):          mean, std, slope, range

Missing-signal handling:
    Subjects with empty IBI files have IBI dropped at load time, so the
    8 HRV features are NaN for their windows. Imputation is downstream's
    job (model pipeline). Same goes for any other missing signal.
"""

import warnings

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy.stats import linregress

# neurokit2 for EDA decomposition + SCR peak detection.
# Wrapped in try/except so the module imports even if not installed,
# but eda features will fail at call time.
try:
    import neurokit2 as nk
    HAS_NEUROKIT = True
except ImportError:
    HAS_NEUROKIT = False


# E4 sampling rates — duplicated from data_loader for self-containment
E4_SAMPLING_RATES = {
    "ACC": 32,
    "BVP": 64,
    "EDA": 4,
    "HR": 1,
    "TEMP": 4,
}

# HRV feature names — used to fill NaN when IBI is missing
HRV_FEATURE_NAMES = [
    "hrv_rmssd", "hrv_sdnn", "hrv_pnn50",
    "hrv_mean_ibi", "hrv_median_ibi",
    "hrv_hr_mean", "hrv_hr_std",
    "hrv_lf_hf_ratio",
]


# ---------- helpers ----------

def _safe_slope(values, fs):
    """Linear regression slope of values vs time. Returns 0 if too few points."""
    if len(values) < 2:
        return 0.0
    t = np.arange(len(values)) / fs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = linregress(t, values)
            return float(res.slope)
        except Exception:
            return 0.0


def _spectral_entropy(values, fs):
    """Normalized spectral entropy via Welch PSD. Hand-rolled to avoid antropy dep."""
    if len(values) < 8:
        return 0.0
    nperseg = min(256, len(values))
    f, psd = scipy_signal.welch(values, fs=fs, nperseg=nperseg)
    psd = psd[psd > 0]
    if len(psd) == 0:
        return 0.0
    psd_norm = psd / psd.sum()
    entropy = -np.sum(psd_norm * np.log2(psd_norm))
    # Normalize by log2 of number of bins
    return float(entropy / np.log2(len(psd_norm))) if len(psd_norm) > 1 else 0.0


def _dominant_freq(values, fs):
    """Frequency of the largest PSD peak."""
    if len(values) < 8:
        return 0.0
    nperseg = min(256, len(values))
    f, psd = scipy_signal.welch(values, fs=fs, nperseg=nperseg)
    if len(psd) == 0:
        return 0.0
    return float(f[np.argmax(psd)])


def _band_power(values, fs, band):
    """Integrated PSD over a frequency band."""
    if len(values) < 8:
        return 0.0
    nperseg = min(256, len(values))
    f, psd = scipy_signal.welch(values, fs=fs, nperseg=nperseg)
    mask = (f >= band[0]) & (f <= band[1])
    if not mask.any():
        return 0.0
    return float(np.trapezoid(psd[mask], f[mask]))


# ---------- HRV from IBI ----------

def extract_hrv_features(ibi_array):
    """
    HRV features from a window's IBI intervals.

    ibi_array shape: (n_intervals, 2) per data_loader.read_e4_ibi columns
    [offset, ibi]. We use the ibi column (last column) in seconds.

    Returns a dict of 8 features. All NaN if fewer than 2 valid intervals
    (HRV is undefined with <2 RR intervals).
    """
    if ibi_array is None or len(ibi_array) < 2:
        return {name: np.nan for name in HRV_FEATURE_NAMES}

    # IBI windows come from create_windows slicing the DataFrame; the last
    # column is the ibi duration in seconds. Defensive: if 1-D somehow, treat
    # whole thing as the durations.
    if ibi_array.ndim == 2:
        ibi_sec = ibi_array[:, -1].astype(float)
    else:
        ibi_sec = ibi_array.astype(float)

    ibi_ms = ibi_sec * 1000.0  # HRV conventions use milliseconds

    # Reject any pathological values (clean_ibi should have done this but
    # be defensive — windows can also contain edge effects)
    valid = (ibi_ms >= 300) & (ibi_ms <= 2000)
    ibi_ms = ibi_ms[valid]
    if len(ibi_ms) < 2:
        return {name: np.nan for name in HRV_FEATURE_NAMES}

    # Time-domain HRV
    diffs = np.diff(ibi_ms)
    rmssd = float(np.sqrt(np.mean(diffs ** 2)))
    sdnn = float(np.std(ibi_ms, ddof=1)) if len(ibi_ms) >= 2 else 0.0
    pnn50 = float(np.mean(np.abs(diffs) > 50.0) * 100.0) if len(diffs) > 0 else 0.0
    mean_ibi = float(np.mean(ibi_ms))
    median_ibi = float(np.median(ibi_ms))

    # Instantaneous HR
    hr_bpm = 60000.0 / ibi_ms
    hr_mean = float(np.mean(hr_bpm))
    hr_std = float(np.std(hr_bpm, ddof=1)) if len(hr_bpm) >= 2 else 0.0

    # LF/HF ratio via Lomb-Scargle (IBI is unevenly sampled in time)
    lf_hf = _lf_hf_ratio(ibi_ms)

    return {
        "hrv_rmssd": rmssd,
        "hrv_sdnn": sdnn,
        "hrv_pnn50": pnn50,
        "hrv_mean_ibi": mean_ibi,
        "hrv_median_ibi": median_ibi,
        "hrv_hr_mean": hr_mean,
        "hrv_hr_std": hr_std,
        "hrv_lf_hf_ratio": lf_hf,
    }


def _lf_hf_ratio(ibi_ms):
    """
    LF (0.04-0.15 Hz) / HF (0.15-0.4 Hz) power ratio via Lomb-Scargle.
    Uses cumulative time as the time axis (IBI tachogram is uneven).
    Returns NaN if either band has zero power or there's not enough data.
    """
    if len(ibi_ms) < 4:
        return np.nan
    # Time axis: cumulative sum of intervals in seconds
    t = np.cumsum(ibi_ms) / 1000.0
    # Center the series (Lomb-Scargle expects zero mean)
    y = ibi_ms - np.mean(ibi_ms)
    # Frequency grid covering LF + HF
    freqs = np.linspace(0.04, 0.4, 64)
    angular_freqs = 2 * np.pi * freqs
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pgram = scipy_signal.lombscargle(t, y, angular_freqs, normalize=True)
    except Exception:
        return np.nan

    lf_mask = (freqs >= 0.04) & (freqs <= 0.15)
    hf_mask = (freqs > 0.15) & (freqs <= 0.4)
    lf_power = float(np.trapezoid(pgram[lf_mask], freqs[lf_mask])) if lf_mask.any() else 0.0
    hf_power = float(np.trapezoid(pgram[hf_mask], freqs[hf_mask])) if hf_mask.any() else 0.0
    if hf_power <= 1e-12:
        return np.nan
    return lf_power / hf_power


# ---------- EDA ----------

def extract_eda_features(eda_values, fs=4):
    """
    10 EDA features. Uses neurokit2 for phasic/tonic decomposition and
    SCR peak detection. If neurokit2 is unavailable, falls back to NaN
    for the SCR-derived features and computes stats from the raw EDA.
    """
    values = np.asarray(eda_values).squeeze().astype(float)
    if values.ndim == 0 or len(values) < 4:
        return {f"eda_{k}": np.nan for k in [
            "mean", "std", "min", "max", "slope",
            "scr_count", "scr_amp_mean", "scr_rise_mean",
            "tonic_mean", "phasic_std",
        ]}

    feats = {
        "eda_mean": float(np.mean(values)),
        "eda_std": float(np.std(values)),
        "eda_min": float(np.min(values)),
        "eda_max": float(np.max(values)),
        "eda_slope": _safe_slope(values, fs),
    }

    # SCR-derived features need neurokit2's cvxEDA decomposition
    if not HAS_NEUROKIT:
        feats.update({
            "eda_scr_count": np.nan,
            "eda_scr_amp_mean": np.nan,
            "eda_scr_rise_mean": np.nan,
            "eda_tonic_mean": np.nan,
            "eda_phasic_std": np.nan,
        })
        return feats

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            decomposed = nk.eda_phasic(values, sampling_rate=fs, method="cvxeda")
        tonic = decomposed["EDA_Tonic"].values
        phasic = decomposed["EDA_Phasic"].values

        # SCR peak detection on the phasic component
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            peaks_info, peaks_meta = nk.eda_peaks(
                phasic, sampling_rate=fs, method="neurokit"
            )

        amps = peaks_meta.get("SCR_Amplitude", np.array([]))
        rise = peaks_meta.get("SCR_RiseTime", np.array([]))
        scr_count = int(np.sum(peaks_info["SCR_Peaks"].values)) if "SCR_Peaks" in peaks_info else 0

        # Drop NaN entries — eda_peaks sometimes returns peaks with
        # undefined amplitude/rise (edge effects at window boundaries).
        # Treat "no valid peaks" as 0, not NaN, to match scr_count semantics.
        amps_valid = np.asarray(amps)[~np.isnan(np.asarray(amps))]
        rise_valid = np.asarray(rise)[~np.isnan(np.asarray(rise))]

        feats.update({
            "eda_scr_count": float(scr_count),
            "eda_scr_amp_mean": float(np.mean(amps_valid)) if len(amps_valid) > 0 else 0.0,
            "eda_scr_rise_mean": float(np.mean(rise_valid)) if len(rise_valid) > 0 else 0.0,
            "eda_tonic_mean": float(np.mean(tonic)),
            "eda_phasic_std": float(np.std(phasic)),
        })
    except Exception:
        # cvxEDA can fail on degenerate (flat / very short) inputs.
        # Fall back to NaN for the SCR features; basic stats are still good.
        feats.update({
            "eda_scr_count": np.nan,
            "eda_scr_amp_mean": np.nan,
            "eda_scr_rise_mean": np.nan,
            "eda_tonic_mean": np.nan,
            "eda_phasic_std": np.nan,
        })

    return feats


# ---------- BVP ----------

def extract_bvp_features(bvp_values, fs=64):
    """6 BVP features: time-domain stats + frequency content."""
    values = np.asarray(bvp_values).squeeze().astype(float)
    if values.ndim == 0 or len(values) < 8:
        return {f"bvp_{k}": np.nan for k in [
            "mean", "std", "spec_entropy", "dom_freq", "lf_power", "hf_power"
        ]}
    return {
        "bvp_mean": float(np.mean(values)),
        "bvp_std": float(np.std(values)),
        "bvp_spec_entropy": _spectral_entropy(values, fs),
        "bvp_dom_freq": _dominant_freq(values, fs),
        "bvp_lf_power": _band_power(values, fs, (0.04, 0.15)),
        "bvp_hf_power": _band_power(values, fs, (0.15, 0.4)),
    }


# ---------- HR ----------

def extract_hr_features(hr_values):
    """4 HR features: time-domain stats. HR is firmware-smoothed, no freq features."""
    values = np.asarray(hr_values).squeeze().astype(float)
    if values.ndim == 0 or len(values) < 1:
        return {f"hr_{k}": np.nan for k in ["mean", "std", "min", "max"]}
    return {
        "hr_mean": float(np.mean(values)),
        "hr_std": float(np.std(values)) if len(values) > 1 else 0.0,
        "hr_min": float(np.min(values)),
        "hr_max": float(np.max(values)),
    }


# ---------- ACC ----------

def extract_acc_features(acc_values, fs=32):
    """
    8 ACC features computed from 3-axis accelerometer. Magnitude features
    use sqrt(x^2 + y^2 + z^2); jerk is the derivative of magnitude; SMA is
    the signal magnitude area (mean of |x| + |y| + |z|).
    """
    arr = np.asarray(acc_values).astype(float)
    if arr.ndim == 1:
        # Pathological: only one axis. Treat as magnitude directly.
        magnitude = np.abs(arr)
        sma = float(np.mean(np.abs(arr)))
    else:
        magnitude = np.sqrt(np.sum(arr ** 2, axis=1))
        sma = float(np.mean(np.sum(np.abs(arr), axis=1)))

    if len(magnitude) < 2:
        return {f"acc_{k}": np.nan for k in [
            "mag_mean", "mag_std", "mag_min", "mag_max",
            "mag_auc", "jerk_std", "dom_freq", "sma"
        ]}

    jerk = np.diff(magnitude) * fs  # per-second derivative
    return {
        "acc_mag_mean": float(np.mean(magnitude)),
        "acc_mag_std": float(np.std(magnitude)),
        "acc_mag_min": float(np.min(magnitude)),
        "acc_mag_max": float(np.max(magnitude)),
        "acc_mag_auc": float(np.trapezoid(magnitude, dx=1.0 / fs)),
        "acc_jerk_std": float(np.std(jerk)),
        "acc_dom_freq": _dominant_freq(magnitude, fs),
        "acc_sma": sma,
    }


# ---------- TEMP ----------

def extract_temp_features(temp_values, fs=4):
    """4 TEMP features: mean, std, slope, range."""
    values = np.asarray(temp_values).squeeze().astype(float)
    if values.ndim == 0 or len(values) < 2:
        return {f"temp_{k}": np.nan for k in ["mean", "std", "slope", "range"]}
    return {
        "temp_mean": float(np.mean(values)),
        "temp_std": float(np.std(values)),
        "temp_slope": _safe_slope(values, fs),
        "temp_range": float(np.max(values) - np.min(values)),
    }


# ---------- Top-level ----------

def extract_features_for_window(window, sampling_rates=None):
    """
    Extract all features for one window dict (from create_windows).

    Args:
        window:           {"signals": {name: np.array}, "label": int,
                           "start_sec": float, "end_sec": float}
        sampling_rates:   dict of signal_name -> fs. If None, uses E4 defaults.

    Returns:
        flat dict of ~40 features, plus label/start_sec/end_sec passthrough.
    """
    if sampling_rates is None:
        sampling_rates = E4_SAMPLING_RATES

    sigs = window["signals"]
    feats = {}

    # HRV
    if "IBI" in sigs:
        feats.update(extract_hrv_features(sigs["IBI"]))
    else:
        feats.update({name: np.nan for name in HRV_FEATURE_NAMES})

    # EDA
    if "EDA" in sigs:
        feats.update(extract_eda_features(sigs["EDA"], fs=sampling_rates.get("EDA", 4)))
    else:
        feats.update({f"eda_{k}": np.nan for k in [
            "mean", "std", "min", "max", "slope",
            "scr_count", "scr_amp_mean", "scr_rise_mean",
            "tonic_mean", "phasic_std",
        ]})

    # BVP
    if "BVP" in sigs:
        feats.update(extract_bvp_features(sigs["BVP"], fs=sampling_rates.get("BVP", 64)))
    else:
        feats.update({f"bvp_{k}": np.nan for k in [
            "mean", "std", "spec_entropy", "dom_freq", "lf_power", "hf_power"
        ]})

    # HR
    if "HR" in sigs:
        feats.update(extract_hr_features(sigs["HR"]))
    else:
        feats.update({f"hr_{k}": np.nan for k in ["mean", "std", "min", "max"]})

    # ACC
    if "ACC" in sigs:
        feats.update(extract_acc_features(sigs["ACC"], fs=sampling_rates.get("ACC", 32)))
    else:
        feats.update({f"acc_{k}": np.nan for k in [
            "mag_mean", "mag_std", "mag_min", "mag_max",
            "mag_auc", "jerk_std", "dom_freq", "sma"
        ]})

    # TEMP
    if "TEMP" in sigs:
        feats.update(extract_temp_features(sigs["TEMP"], fs=sampling_rates.get("TEMP", 4)))
    else:
        feats.update({f"temp_{k}": np.nan for k in ["mean", "std", "slope", "range"]})

    # Metadata passthrough
    feats["label"] = window["label"]
    feats["start_sec"] = window["start_sec"]
    feats["end_sec"] = window["end_sec"]

    return feats


def extract_features(windows, subject_id=None, dataset_name=None,
                     sampling_rates=None):
    """
    Extract features for a list of windows. Returns a DataFrame with one
    row per window. Includes subject_id and dataset columns if provided.

    Args:
        windows:          list of window dicts from create_windows
        subject_id:       optional, added as a column
        dataset_name:     optional, added as a column
        sampling_rates:   optional, falls back to E4 defaults

    Returns:
        pd.DataFrame, shape (n_windows, ~43)
    """
    if not windows:
        return pd.DataFrame()

    rows = []
    for i, w in enumerate(windows):
        feats = extract_features_for_window(w, sampling_rates=sampling_rates)
        feats["window_idx"] = i
        if subject_id is not None:
            feats["subject_id"] = subject_id
        if dataset_name is not None:
            feats["dataset"] = dataset_name
        rows.append(feats)

    df = pd.DataFrame(rows)

    # Reorder: metadata first, then features
    meta_cols = [c for c in
                 ["dataset", "subject_id", "window_idx", "start_sec", "end_sec", "label"]
                 if c in df.columns]
    feat_cols = [c for c in df.columns if c not in meta_cols]
    return df[meta_cols + feat_cols]