"""
Extract stress/non-stress labels from WESAD .pkl files.

WESAD .pkl structure:
  data = {
      'subject': 'SX',
      'label':   np.array of shape (N,), int, sampled at 700 Hz
                 0 = transient/undefined
                 1 = baseline
                 2 = stress
                 3 = amusement
                 4 = meditation
                 5,6,7 = ignore
      'signal':  {'chest': {...}, 'wrist': {...}}  -- we don't touch this
  }

For each subject we:
  1. Open SX.pkl from inside WESAD.zip (no full extraction)
  2. Read the label array, discard the rest
  3. Find contiguous runs of label == 1 and label == 2
  4. Merge same-label runs separated by gaps shorter than MERGE_GAP_SEC
  5. Emit (start_sec, end_sec, label) rows where
     - label 1 (WESAD baseline) -> 0 (non-stress, our convention)
     - label 2 (WESAD stress)   -> 1 (stress)
  6. Write labels/wesad/{subject}.csv

Configure paths via the variables at the top of main().
Run from project root:
    python scripts/extract_wesad_labels.py
"""
import io
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


LABEL_SAMPLING_RATE = 700  # Hz, fixed by WESAD protocol (chest device rate)

# WESAD label values we keep, and their mapping to our binary convention
WESAD_TO_BINARY = {
    1: 0,  # baseline -> non-stress
    2: 1,  # stress   -> stress
}

# Two same-label runs separated by a gap shorter than this are merged into one
MERGE_GAP_SEC = 5.0

# WESAD subjects (S12 missing in the original dataset)
WESAD_SUBJECTS = [f"S{i}" for i in range(2, 18) if i != 12]


def find_runs(label_array, target_label):
    """
    Find contiguous runs of `target_label` in `label_array`.

    Returns list of (start_idx, end_idx) tuples, where end_idx is exclusive.
    """
    mask = (label_array == target_label).astype(np.int8)
    if mask.sum() == 0:
        return []

    # Detect transitions: prepend/append 0 so edges are caught
    padded = np.concatenate(([0], mask, [0]))
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def merge_close_runs(runs, max_gap_samples):
    """
    Merge consecutive runs whose gap is <= max_gap_samples.

    Args:
        runs:             list of (start, end) tuples, sorted by start
        max_gap_samples:  int, gap threshold in samples

    Returns:
        list of (start, end) tuples
    """
    if not runs:
        return []

    merged = [list(runs[0])]
    for start, end in runs[1:]:
        prev_end = merged[-1][1]
        gap = start - prev_end
        if gap <= max_gap_samples:
            merged[-1][1] = end  # extend previous run
        else:
            merged.append([start, end])
    return [tuple(r) for r in merged]


def extract_phases_for_subject(label_array):
    """
    Extract (start_sec, end_sec, binary_label) phases from a 700Hz label array.

    Returns a list of rows sorted by start_sec.
    """
    max_gap_samples = int(MERGE_GAP_SEC * LABEL_SAMPLING_RATE)
    rows = []

    for wesad_label, binary_label in WESAD_TO_BINARY.items():
        runs = find_runs(label_array, wesad_label)
        runs = merge_close_runs(runs, max_gap_samples)
        for start_idx, end_idx in runs:
            start_sec = start_idx / LABEL_SAMPLING_RATE
            end_sec = end_idx / LABEL_SAMPLING_RATE
            rows.append((start_sec, end_sec, binary_label))

    rows.sort(key=lambda r: r[0])
    return rows


def load_label_array_from_zip(wesad_zip_path, subject_id):
    """
    Read SX.pkl from inside WESAD.zip and return just the label array.
    Signals are discarded immediately to keep memory low.
    """
    inner_path = f"WESAD/{subject_id}/{subject_id}.pkl"
    with zipfile.ZipFile(wesad_zip_path, "r") as outer:
        with outer.open(inner_path) as pkl_file:
            data = pickle.load(pkl_file, encoding="latin1")

    labels = np.asarray(data["label"])
    del data  # release signals
    return labels


def process_subject(wesad_zip_path, subject_id, output_dir):
    """Extract labels for one subject and write CSV."""
    labels = load_label_array_from_zip(wesad_zip_path, subject_id)
    phases = extract_phases_for_subject(labels)

    df = pd.DataFrame(phases, columns=["start_sec", "end_sec", "label"])
    output_path = output_dir / f"{subject_id}.csv"
    df.to_csv(output_path, index=False)

    n_nonstress = (df["label"] == 0).sum()
    n_stress = (df["label"] == 1).sum()
    total_duration = labels.shape[0] / LABEL_SAMPLING_RATE
    print(
        f"  {subject_id}: {total_duration/60:.1f} min total, "
        f"{n_nonstress} non-stress phase(s), {n_stress} stress phase(s) -> {output_path}"
    )


def main():
    drive_root = "E:/Hifsa/AML/Project/datasets"
    output_dir = Path("labels/wesad")
    subjects = WESAD_SUBJECTS  # set to e.g. ["S2"] to test on one subject
    #subjects = ["S2"]

    wesad_zip_path = Path(drive_root) / "WESAD.zip"
    if not wesad_zip_path.exists():
        raise FileNotFoundError(f"WESAD zip not found: {wesad_zip_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting labels for {len(subjects)} subject(s) from {wesad_zip_path}")
    print(f"Output: {output_dir}/")

    for subject_id in subjects:
        try:
            process_subject(wesad_zip_path, subject_id, output_dir)
        except KeyError as e:
            print(f"  {subject_id}: SKIPPED (pkl not found in zip: {e})")
        except Exception as e:
            print(f"  {subject_id}: ERROR -- {type(e).__name__}: {e}")

    print("Done.")


if __name__ == "__main__":
    main()