"""
EDA summary builder.

build_dataset_summary() walks each dataset, loads every subject (or session for
Nurse), runs windowing with the right label source, and returns a dict of
per-subject records plus aggregate stats. Used by 01_eda.ipynb to produce
figures 2-6 and the dataset characterization table.

Per-subject record shape:
    {
        "id":              str,    # subject/session id
        "nurse_id":        str|None,  # only for nurse — nurse-level grouping
        "duration_sec":    float,
        "stress_sec":      float,
        "non_stress_sec":  float,
        "unlabeled_sec":   float,
        "n_windows":       int,
        "stress_windows":  int,
        "non_stress_windows": int,
        "signals_present": dict[str, bool],   # ACC/BVP/EDA/HR/IBI/TEMP
        "excluded":        bool,
        "exclude_reason":  str|None,
    }
"""

from pathlib import Path

import pandas as pd

from src.data_loader import load_subject, list_subjects, SIGNAL_NAMES
from src.labeling import get_campanella_labels, get_wesad_labels
from src.windowing import create_windows


def _duration_from_signals(signals):
    """Recording duration in seconds from the shortest signal."""
    if not signals:
        return 0.0
    durations = [
        (sig.index[-1] - sig.index[0]).total_seconds()
        for sig in signals.values() if len(sig) > 0
    ]
    return min(durations) if durations else 0.0


def _labeled_duration(phases, label):
    """Total seconds covered by phases with this label."""
    return sum(end - start for start, end, lbl in phases if lbl == label)


def _signal_presence(signals):
    """Map signal name -> True if present and non-empty."""
    return {
        name: (name in signals and len(signals[name]) > 0)
        for name in SIGNAL_NAMES
    }


def _load_nurse_phases(session_id, labels_root):
    """Load nurse phases from the committed labels CSV; empty if no file."""
    path = Path(labels_root) / "nurse" / f"{session_id}.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return list(df.itertuples(index=False, name=None))


def _build_record(subject_id, signals, phases, duration_sec,
                  nurse_id=None, excluded=False, exclude_reason=None):
    """Common record-building shared across datasets."""
    if excluded or not signals or duration_sec == 0:
        return {
            "id": subject_id,
            "nurse_id": nurse_id,
            "duration_sec": duration_sec,
            "stress_sec": 0.0,
            "non_stress_sec": 0.0,
            "unlabeled_sec": duration_sec,
            "n_windows": 0,
            "stress_windows": 0,
            "non_stress_windows": 0,
            "signals_present": _signal_presence(signals),
            "excluded": True,
            "exclude_reason": exclude_reason or "no usable data",
        }

    stress_sec = _labeled_duration(phases, 1)
    non_stress_sec = _labeled_duration(phases, 0)
    unlabeled_sec = max(0.0, duration_sec - stress_sec - non_stress_sec)

    windows = create_windows(signals, phases, total_duration=duration_sec)
    stress_w = sum(1 for w in windows if w["label"] == 1)
    non_stress_w = sum(1 for w in windows if w["label"] == 0)

    return {
        "id": subject_id,
        "nurse_id": nurse_id,
        "duration_sec": duration_sec,
        "stress_sec": stress_sec,
        "non_stress_sec": non_stress_sec,
        "unlabeled_sec": unlabeled_sec,
        "n_windows": len(windows),
        "stress_windows": stress_w,
        "non_stress_windows": non_stress_w,
        "signals_present": _signal_presence(signals),
        "excluded": False,
        "exclude_reason": None,
    }


def _summarize_campanella(data_root):
    records = []
    for subject_id in list_subjects("campanella", data_root):
        try:
            signals, _ = load_subject("campanella", subject_id, data_root)
            duration = _duration_from_signals(signals)
            try:
                phases = get_campanella_labels(duration)
            except ValueError as e:
                records.append(_build_record(
                    subject_id, signals, [], duration,
                    excluded=True, exclude_reason=f"protocol incomplete ({duration/60:.1f} min)"
                ))
                continue
            records.append(_build_record(subject_id, signals, phases, duration))
        except Exception as e:
            records.append(_build_record(
                subject_id, {}, [], 0,
                excluded=True, exclude_reason=f"load error: {type(e).__name__}"
            ))
    return records


def _summarize_wesad(data_root, labels_root):
    records = []
    for subject_id in list_subjects("wesad", data_root):
        try:
            signals, _ = load_subject("wesad", subject_id, data_root)
            duration = _duration_from_signals(signals)
            try:
                phases = get_wesad_labels(subject_id, labels_root=f"{labels_root}/wesad")
            except FileNotFoundError:
                records.append(_build_record(
                    subject_id, signals, [], duration,
                    excluded=True, exclude_reason="no labels file"
                ))
                continue
            records.append(_build_record(subject_id, signals, phases, duration))
        except Exception as e:
            records.append(_build_record(
                subject_id, {}, [], 0,
                excluded=True, exclude_reason=f"load error: {type(e).__name__}"
            ))
    return records


def _summarize_nurse(data_root, labels_root):
    """
    For Nurse, "subjects" in list_subjects() are actually session folders
    (e.g., '83_1604630543'). We treat each session as one record and also
    tag it with the nurse_id for grouping.

    Sessions without a labels CSV are still recorded — they show up in
    duration distributions and signal-availability stats but contribute
    zero windows.
    """
    records = []
    for session_id in list_subjects("nurse", data_root):
        nurse_id = session_id.split("_")[0]
        try:
            signals, _ = load_subject("nurse", session_id, data_root)
            duration = _duration_from_signals(signals)
            phases = _load_nurse_phases(session_id, labels_root)

            if not phases:
                records.append(_build_record(
                    session_id, signals, [], duration,
                    nurse_id=nurse_id,
                    excluded=True, exclude_reason="no labeled surveys in session"
                ))
                continue
            records.append(_build_record(
                session_id, signals, phases, duration, nurse_id=nurse_id
            ))
        except Exception as e:
            records.append(_build_record(
                session_id, {}, [], 0,
                nurse_id=nurse_id,
                excluded=True, exclude_reason=f"load error: {type(e).__name__}"
            ))
    return records


def build_dataset_summary(data_root="data_extracted", labels_root="labels"):
    """
    Walk all three datasets and return per-subject records.

    Returns:
        dict: {"campanella": [records], "wesad": [records], "nurse": [records]}
    """
    print("Building summary for campanella...")
    campanella = _summarize_campanella(data_root)
    print(f"  {len(campanella)} subjects "
          f"({sum(1 for r in campanella if not r['excluded'])} usable)")

    print("Building summary for wesad...")
    wesad = _summarize_wesad(data_root, labels_root)
    print(f"  {len(wesad)} subjects "
          f"({sum(1 for r in wesad if not r['excluded'])} usable)")

    print("Building summary for nurse...")
    nurse = _summarize_nurse(data_root, labels_root)
    n_sessions_used = sum(1 for r in nurse if not r['excluded'])
    n_nurses_used = len({r['nurse_id'] for r in nurse if not r['excluded']})
    print(f"  {len(nurse)} sessions across {len({r['nurse_id'] for r in nurse})} nurses "
          f"({n_sessions_used} sessions, {n_nurses_used} nurses usable)")

    return {"campanella": campanella, "wesad": wesad, "nurse": nurse}


def summary_to_dataframe(summary):
    """Flatten the summary dict into one DataFrame for analysis/saving."""
    rows = []
    for dataset_name, records in summary.items():
        for r in records:
            row = {"dataset": dataset_name, **r}
            # Flatten signals_present into separate columns
            for sig, present in row.pop("signals_present").items():
                row[f"has_{sig}"] = present
            rows.append(row)
    return pd.DataFrame(rows)


def build_characterization_table(summary):
    """
    Build the dataset-level characterization table from the per-subject summary.
    Returns a DataFrame indexed by dataset with columns suitable for the paper.
    """
    rows = []
    for dataset_name, records in summary.items():
        kept = [r for r in records if not r["excluded"]]
        excluded = [r for r in records if r["excluded"]]

        if not kept:
            rows.append({"dataset": dataset_name, "n_subjects_kept": 0})
            continue

        durations_min = [r["duration_sec"] / 60 for r in kept]
        stress_min = [r["stress_sec"] / 60 for r in kept]
        non_stress_min = [r["non_stress_sec"] / 60 for r in kept]
        unlabeled_pct = [
            100 * r["unlabeled_sec"] / r["duration_sec"]
            for r in kept if r["duration_sec"] > 0
        ]

        n_windows = sum(r["n_windows"] for r in kept)
        n_stress_w = sum(r["stress_windows"] for r in kept)
        n_non_stress_w = sum(r["non_stress_windows"] for r in kept)

        ratio = (n_stress_w / n_non_stress_w) if n_non_stress_w > 0 else float("inf")

        rows.append({
            "dataset": dataset_name,
            "n_subjects_kept": len(kept),
            "n_subjects_excluded": len(excluded),
            "duration_min_mean": _mean(durations_min),
            "duration_min_std": _std(durations_min),
            "duration_min_min": min(durations_min),
            "duration_min_max": max(durations_min),
            "stress_min_mean": _mean(stress_min),
            "non_stress_min_mean": _mean(non_stress_min),
            "unlabeled_pct_mean": _mean(unlabeled_pct),
            "n_windows_total": n_windows,
            "n_stress_windows": n_stress_w,
            "n_non_stress_windows": n_non_stress_w,
            "stress_to_non_stress_ratio": ratio,
        })

    return pd.DataFrame(rows).set_index("dataset")


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5