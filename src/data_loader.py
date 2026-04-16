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

def read_e4_csv(filepath, signal_name):
    """Read a single Empatica E4 CSV file and return a time-indexed DataFrame."""

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, "r") as f:
        line1 = f.readline().strip()
        line2 = f.readline().strip()

    first_val = float(line1.split(",")[0].split()[0]) 
    
    if first_val > 1_000_000_000:
        # WESAD/Nurse format: line1=timestamp, line2=sampling_rate, skip 2 rows
        start_timestamp = first_val
        sampling_rate = float(line2.split(",")[0].split()[0])
        skip_rows = 2
    else:
        # Campanella format: no timestamp, use known sampling rate
        start_timestamp = 0
        sampling_rate = E4_SAMPLING_RATES[signal_name]
        skip_rows = 0

    df = pd.read_csv(filepath, skiprows=2, header=None)

    n_samples = len(df)
    timestamps = start_timestamp + np.arange(n_samples) / sampling_rate
    df.index = pd.to_datetime(timestamps, unit="s", utc=True)
    df.index.name = "timestamp"

    return df, sampling_rate


def read_e4_ibi(filepath):
    """
    Read IBI file — different format from other E4 files.
    Line 1: start timestamp
    Remaining lines: time_offset, ibi_duration
    """
    # Handle empty files
    if os.path.getsize(filepath) == 0:
        raise ValueError(f"IBI file is empty: {filepath}")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, "r") as f:
        line1 = f.readline().strip()

    first_val = float(line1.split(",")[0].split()[0]) 
    
    if first_val > 1_000_000_000:
        # WESAD/Nurse format: line1=timestamp, line2=sampling_rate, skip 2 rows
        start_timestamp = first_val
        skip_rows = 1
    else:
        # Campanella format: no timestamp, use known sampling rate
        start_timestamp = 0
        skip_rows = 0

    df = pd.read_csv(filepath, skiprows=1, header=None, names=["offset", "ibi"])

    timestamps = pd.to_datetime(
        start_timestamp + df["offset"].values, unit="s", utc=True
    )
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

    # Test with one subject — update this path to match your setup
    # signals, rates = load_subject(
    #     dataset_name="campanella",
    #     subject_id="subject_01",
    #     data_root="data_extracted"
    # )

    # for name, df in signals.items():
    #     print(f"\n{name}:")
    #     print(f"  Shape: {df.shape}")
    #     print(f"  Time: {df.index[0]} → {df.index[-1]}")
    #     print(df.head(3))