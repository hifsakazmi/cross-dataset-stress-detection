import os
import pandas as pd
import numpy as np
from pathlib import Path

SIGNAL_NAMES = ["ACC", "BVP", "EDA", "HR", "IBI", "TEMP"]

def read_e4_csv(filepath):
    """Read a single Empatica E4 CSV file and return a time-indexed DataFrame."""

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, "r") as f:
        line1 = f.readline().strip()
        line2 = f.readline().strip()

    start_timestamp = float(line1.split(",")[0])
    sampling_rate = float(line2.split(",")[0])

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

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, "r") as f:
        line1 = f.readline().strip()

    start_timestamp = float(line1.split(",")[0])

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
                df, sr = read_e4_csv(filepath)
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


def load_dataset(dataset_name, data_root="data", subjects=None):
    """Load all subjects from a dataset."""

    if subjects is None:
        subjects = list_subjects(dataset_name, data_root)

    print(f"Loading {dataset_name} ({len(subjects)} subjects)...")

    dataset = {}
    for subject_id in subjects:
        signals, sampling_rates = load_subject(dataset_name, subject_id, data_root)
        dataset[subject_id] = {"signals": signals, "sampling_rates": sampling_rates}

    print(f"Done. Loaded {len(dataset)} subjects.\n")
    return dataset


if __name__ == "__main__":
    # Test with one subject — update this path to match your setup
    signals, rates = load_subject(
        dataset_name="nurse",
        subject_id="5C/5C_1586886626",
        data_root="E:/Hifsa/AML/Project/Nurses-dataset/Stress_dataset"
    )

    for name, df in signals.items():
        print(f"\n{name}:")
        print(f"  Shape: {df.shape}")
        print(f"  Time: {df.index[0]} → {df.index[-1]}")
        print(df.head(3))