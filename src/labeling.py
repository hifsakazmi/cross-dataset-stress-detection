"""
Label extraction for each dataset.

Each dataset has a different way of determining stress/non-stress periods:
- Campanella: Fixed protocol timing (Task 4 variable)
- WESAD: Per-subject quest.csv files
- Nurse: Self-reported survey responses

This module returns labels as (start_sec, end_sec, label) tuples,
where label is 0 (non-stress) or 1 (stress).
"""


def get_campanella_labels(total_duration_sec):
    """
    Compute stress/non-stress time ranges for a Campanella subject.

    Protocol (from Campanella et al., Sensors 2023):
      Rest 1 → Task 1 → Rest 2 → Task 2 → Rest 3 → Task 3 →
      Rest 4 → Task 4 (variable) → Rest 5 → Task 5 → Rest 6

    Args:
        total_duration_sec: Total recording duration in seconds.

    Returns:
        List of (start_sec, end_sec, label) tuples covering the full recording.
    """
    # Minimum expected duration (without Task 4): ~32 minutes
    MIN_DURATION = 1620 + 300  # fixed phases + Rest5+Task5+Rest6

    if total_duration_sec < MIN_DURATION:
        raise ValueError(
            f"Recording too short ({total_duration_sec}s) — "
            f"expected at least {MIN_DURATION}s for full protocol."
        )

    # Fixed phases from the start
    phases = [
        (0, 180, 0),        # Rest 1       (3 min)
        (180, 780, 1),      # Task 1       (10 min) — Lego, no instructions
        (780, 900, 0),      # Rest 2       (2 min)
        (900, 1200, 1),     # Task 2       (5 min)  — Lego, with instructions
        (1200, 1320, 0),    # Rest 3       (2 min)
        (1320, 1500, 1),    # Task 3       (3 min)  — Large Lego + counting
        (1500, 1620, 0),    # Rest 4       (2 min)
    ]

    # Task 4 is variable; work backward from end of recording
    task4_start = 1620
    task4_end = total_duration_sec - 300  # Rest5 + Task5 + Rest6 = 5 min

    phases.append((task4_start, task4_end, 1))              # Task 4 — math
    phases.append((task4_end, task4_end + 120, 0))          # Rest 5
    phases.append((task4_end + 120, task4_end + 180, 1))    # Task 5 — presentation
    phases.append((task4_end + 180, total_duration_sec, 0)) # Rest 6

    return phases


def get_label_for_time(time_sec, phases):
    """
    Return the label for a given time within a recording.

    Args:
        time_sec: Time in seconds from recording start.
        phases:   List of (start_sec, end_sec, label) tuples.

    Returns:
        Label (0 or 1), or None if time doesn't fall in any phase.
    """
    for start, end, label in phases:
        if start <= time_sec < end:
            return label
    return None