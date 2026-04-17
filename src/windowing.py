"""
Windowing module: segments continuous signals into fixed-length windows
with labels.
"""

import numpy as np
import pandas as pd
from src.labeling import get_label_for_time


def create_windows(signals, phases, total_duration=0, window_size_sec=60, overlap_sec=30):
    """
    Segment a subject's signals into overlapping windows with labels.

    Args:
        signals:          Dict of signal_name -> DataFrame (from data_loader)
        phases:           List of (start_sec, end_sec, label) tuples
        window_size_sec:  Window length in seconds
        overlap_sec:      Overlap between consecutive windows in seconds

    Returns:
        List of dicts, each containing:
            - "signals": dict of signal_name -> np.array (values in this window)
            - "label": int (0 or 1)
            - "start_sec": float
            - "end_sec": float
    """
    # Pick any signal to determine recording start time and total duration
    ref_signal = list(signals.values())[0]
    recording_start = ref_signal.index[0]
    # total_duration = (ref_signal.index[-1] - recording_start).total_seconds()

    step_sec = window_size_sec - overlap_sec
    windows = []

    current_start = 0.0
    while current_start + window_size_sec <= total_duration:
        current_end = current_start + window_size_sec
        center_time = current_start + window_size_sec / 2

        # Get label at window center
        label = get_label_for_time(center_time, phases)

        # Skip windows that don't fall in any labeled phase
        if label is None:
            current_start += step_sec
            continue

        # Slice each signal for this window
        window_start_ts = recording_start + pd.Timedelta(seconds=current_start)
        window_end_ts = recording_start + pd.Timedelta(seconds=current_end)

        window_signals = {}
        for sig_name, sig_df in signals.items():
            chunk = sig_df.loc[window_start_ts:window_end_ts]
            if len(chunk) > 0:
                window_signals[sig_name] = chunk.values

        # Only keep window if we got at least one signal
        if window_signals:
            windows.append({
                "signals": window_signals,
                "label": label,
                "start_sec": current_start,
                "end_sec": current_end,
            })

        current_start += step_sec

    return windows