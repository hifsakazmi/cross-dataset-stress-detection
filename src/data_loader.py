import os
import pandas as pd
import numpy as np
from pathlib import Path

SIGNAL_NAMES = ["ACC", "BVP", "EDA", "HR", "IBI", "TEMP"]

E4_SAMPLING_RATES = {
    "ACC": 32,
    "BVP": 64,
    "EDA": 4,
    "HR": 1,
    "TEMP": 4,
}


def _clean_corrupt_numerics(df):
    """
    Some Campanella CSVs contain stray malformed values like '1.038.145' mixed
    into otherwise-normal columns of small decimals (e.g. EDA values around
    0.5). Investigation: these multi-dot values cannot be physically valid
    (1,038,145 µS is impossible for EDA, which is bounded ~0-30 µS), and they
    appear sparsely (~3% of samples in affected files). They are corruption,
    not data — likely an export bug in the dataset.

    Strategy:
      1. For object/string columns, parse numerics; anything with 2+ dots or
         any other non-numeric junk becomes NaN.
      2. Forward-fill the resulting NaN values within the column. EDA, BVP,
         TEMP all vary slowly relative to 32-64 Hz sampling, so carrying the
         previous valid value across an isolated bad sample introduces
         negligible distortion. Length and time alignment are preserved,
         which matters because downstream filtfilt and windowing both assume
         uniform sampling.
      3. If the first row(s) are NaN (no prior value to fill from),
         backward-fill from the next valid value.

    Already-numeric columns pass through untouched, so WESAD/Nurse files are
    unaffected.
    """
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        # Coerce to numeric; any value that isn't a plain int/float (multi-dot
        # strings, stray text, etc.) becomes NaN.
        coerced = pd.to_numeric(df[col].astype(str), errors="coerce")
        # Forward-fill, then backward-fill for any leading NaN.
        coerced = coerced.ffill().bfill()
        df[col] = coerced
    return df


def read_e4_csv(filepath, signal_name):
    """
    Read a single Empatica E4 CSV file and return a time-indexed DataFrame.

    Detects two header formats:
      - WESAD/Nurse: line 1 = Unix timestamp, line 2 = sampling rate
      - Campanella: no header; use known E4 sampling rate

    Important: the returned DataFrame is always indexed relative to t=0,
    not the original Unix timestamp. This is intentional. Within the
    Campanella dataset, some subjects have a mix of E4-header files
    (ACC/TEMP/IBI) and headerless files (BVP/EDA/HR) in the same folder
    — likely due to inconsistent export tooling. If we honored the
    embedded timestamps, those signals would not share an epoch and
    windowing (which slices by absolute time) would silently drop the
    headerless signals. Forcing every file to t=0 keeps signals aligned
    within a subject. No downstream code uses the absolute Unix time;
    label files and windowing both operate in relative seconds.
    """

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, "r") as f:
        line1 = f.readline().strip()
        line2 = f.readline().strip()

    first_val = float(line1.split(",")[0].split()[0])

    if first_val > 1_000_000_000:
        # E4-header format: line1=timestamp, line2=sampling_rate, data from row 3
        sampling_rate = float(line2.split(",")[0].split()[0])
        skip_rows = 2
    else:
        # No-header format (Campanella): raw data from row 1
        sampling_rate = E4_SAMPLING_RATES[signal_name]
        skip_rows = 0

    df = pd.read_csv(filepath, skiprows=skip_rows, header=None)

    # Coerce any object-dtype columns (handles stray malformed values in
    # some Campanella files; see _clean_corrupt_numerics docstring).
    df = _clean_corrupt_numerics(df)

    # All signals start at t=0 within a subject (see docstring above).
    n_samples = len(df)
    timestamps = np.arange(n_samples) / sampling_rate
    df.index = pd.to_datetime(timestamps, unit="s", utc=True)
    df.index.name = "timestamp"

    return df, sampling_rate


def read_e4_ibi(filepath):
    """
    Read IBI file — different format from other E4 files.
    Line 1: start timestamp (or "IBI" header text in some Campanella files)
    Remaining lines: time_offset, ibi_duration

    Index is relative to t=0, matching read_e4_csv. See that function's
    docstring for why we don't honor the absolute Unix timestamp.
    """
    # Handle empty files
    if os.path.getsize(filepath) == 0:
        raise ValueError(f"IBI file is empty: {filepath}")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(filepath, skiprows=1, header=None, names=["offset", "ibi"])

    # Defensive coercion in case any IBI files have stray text values
    df = _clean_corrupt_numerics(df)

    # Index by offset directly — these are already relative seconds.
    timestamps = pd.to_datetime(df["offset"].values, unit="s", utc=True)
    df.index = timestamps
    df.index.name = "timestamp"

    return df


def load_subject(dataset_name, subject_id, data_root="data"):
    """Load all E4 signals for a single subject."""

    subject_path = os.path.join(data_root, dataset_name, subject_id)

    if not os.path.exists(subject_path):
        raise FileNotFoundError(f"Subject folder not found: {subject_path}")

    signals = {}
    sampling_rates = {}

    for signal_name in SIGNAL_NAMES:
        filepath = os.path.join(subject_path, f"{signal_name}.csv")

        if not os.path.exists(filepath):
            print(f"  Warning: {signal_name}.csv not found for {subject_id}, skipping.")
            continue

        try:
            if signal_name == "IBI":
                signals["IBI"] = read_e4_ibi(filepath)
                sampling_rates["IBI"] = None
            else:
                df, sr = read_e4_csv(filepath, signal_name=signal_name)
                signals[signal_name] = df
                sampling_rates[signal_name] = sr
        except Exception as e:
            print(f"  Error reading {signal_name}.csv for {subject_id}: {e}")
            continue

    print(f"  Loaded {subject_id}: {list(signals.keys())}")
    return signals, sampling_rates


def list_subjects(dataset_name, data_root="data"):
    """List all subject folders that contain CSV files."""

    dataset_path = os.path.join(data_root, dataset_name)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset folder not found: {dataset_path}")

    subjects = []
    for item in sorted(os.listdir(dataset_path)):
        item_path = os.path.join(dataset_path, item)
        if os.path.isdir(item_path):
            csv_files = [f for f in os.listdir(item_path) if f.endswith(".csv")]
            if csv_files:
                subjects.append(item)

    return subjects


def load_dataset(dataset_name, data_root="data"):
    """Load all subjects from a dataset."""
    subjects = list_subjects(dataset_name, data_root)

    print(f"Loading {dataset_name} ({len(subjects)} subjects)...")

    dataset = {}
    for subject_id in subjects:
        signals, sampling_rates = load_subject(dataset_name, subject_id, data_root)
        dataset[subject_id] = {"signals": signals, "sampling_rates": sampling_rates}

    print(f"Done. Loaded {len(dataset)} subjects.\n")
    return dataset


if __name__ == "__main__":
    data_path = "data_extracted"
    load_dataset('campanella', data_path)
    load_dataset('nurse', data_path)
    load_dataset('wesad', data_path)