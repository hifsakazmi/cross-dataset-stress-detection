"""
Extract stress/non-stress labels from SurveyResults.xlsx for the Nurse dataset.

The nurse dataset spans many short E4 recording sessions per nurse (~40/nurse).
Each surveyed event in SurveyResults.xlsx has a Start time and End time on a
specific date; we map those to whichever session contains them, then emit
(start_sec, end_sec, label) rows relative to that session's recording start.

Stress threshold (matches Hosseini et al. 2022 binary protocol):
  Stress level 0 -> 0 (non-stress)
  Stress level 1 -> SKIPPED (medium, ambiguous)
  Stress level 2 -> 1 (stress)
  na             -> SKIPPED

For each session folder in data_extracted/nurse/:
  1. Read session start_ts from EDA.csv line 1 (Unix timestamp)
  2. Compute session end_ts from (num_samples / 4 Hz) + start_ts
  3. Find surveys belonging to the same nurse whose
     [survey_start_ts, survey_end_ts] falls within [start_ts, end_ts]
  4. Convert to (start_sec, end_sec, label) relative to start_ts
  5. If any survey matches, write labels/nurse/{session_folder}.csv

Sessions without matching surveys produce no CSV.

Configure paths via the variables at the top of main().
Run from project root:
    python scripts/extract_nurse_labels.py
"""

import csv
from pathlib import Path

import pandas as pd


STRESS_LEVEL_TO_BINARY = {
    0: 0,  # low  -> non-stress
    2: 1,  # high -> stress
    # 1 (medium) and "na" intentionally skipped
}

EDA_SAMPLING_RATE = 4  # Hz, fixed by E4 spec


def parse_surveys(survey_xlsx_path):
    """
    Read SurveyResults.xlsx and return a DataFrame with usable surveys only.

    Returns columns:
        nurse_id (str), survey_start_ts (float), survey_end_ts (float), label (int)

    Where survey_start_ts and survey_end_ts are Unix timestamps in seconds.
    """
    df = pd.read_excel(survey_xlsx_path)

    # The Stress level column has int 0,1,2 and string "na" mixed in.
    # Coerce to numeric; non-numeric becomes NaN; then drop unusable rows.
    df["stress_numeric"] = pd.to_numeric(df["Stress level"], errors="coerce")
    df = df[df["stress_numeric"].isin(STRESS_LEVEL_TO_BINARY.keys())].copy()
    df["label"] = df["stress_numeric"].astype(int).map(STRESS_LEVEL_TO_BINARY)

    # Build Unix timestamps. The Excel `Start time` and `End time` are
    # datetime.time objects (or fractions of a day). Combine with `date`.
    def combine_dt(row, time_col):
        date_val = row["date"]
        time_val = row[time_col]
        # If it's already a datetime/Timestamp, just take it
        if hasattr(time_val, "hour"):
            # pandas Timestamp or datetime.time
            if hasattr(time_val, "year"):
                # Full Timestamp - use as is
                return pd.Timestamp(time_val)
            else:
                # time only - combine with date
                return pd.Timestamp.combine(date_val.date(), time_val)
        # Fallback: numeric fraction of day
        return pd.Timestamp(date_val) + pd.Timedelta(days=float(time_val))

    df["start_dt"] = df.apply(lambda r: combine_dt(r, "Start time"), axis=1)
    df["end_dt"] = df.apply(lambda r: combine_dt(r, "End time"), axis=1)

    # The surveys are local time. The E4 timestamps in the CSVs are Unix
    # (UTC) seconds. We assume the survey times are in the same timezone
    # the device was configured for (typically the local hospital time,
    # written to the device clock). If timezone offsets matter we'd need
    # the original device timezone -- the Dryad metadata doesn't specify.
    # For now: treat survey times as UTC-naive seconds since epoch.
    # Normalize to nanosecond precision first so // 10**9 is correct
    # regardless of whether the Excel column came back as us / ms / ns.
    df["survey_start_ts"] = (
        df["start_dt"].astype("datetime64[ns]").astype("int64") // 10**9
    )
    df["survey_end_ts"] = (
        df["end_dt"].astype("datetime64[ns]").astype("int64") // 10**9
    )

    df["nurse_id"] = df["ID"].astype(str)

    return df[["nurse_id", "survey_start_ts", "survey_end_ts", "label"]].reset_index(drop=True)


def get_session_time_range(session_folder):
    """
    Read EDA.csv from a session folder to get (start_ts, end_ts) in Unix seconds.

    EDA.csv line 1 = start_ts; line 2 = sampling rate; lines 3+ = data.
    Session end = start_ts + (num_samples / 4 Hz).
    """
    eda_path = session_folder / "EDA.csv"
    if not eda_path.exists():
        return None

    with open(eda_path, "r") as f:
        start_ts = float(f.readline().strip().split(",")[0])
        _sampling_rate = f.readline()  # skip
        num_samples = sum(1 for _ in f)

    if num_samples == 0:
        return None

    end_ts = start_ts + num_samples / EDA_SAMPLING_RATE
    return start_ts, end_ts


def extract_nurse_id_from_folder(folder_name):
    """e.g. '5C_1586886626' -> '5C'."""
    return folder_name.split("_")[0]


def process_session(session_folder, surveys_df, output_dir):
    """
    Match surveys to one session folder, write a label CSV if any match.
    Returns the number of survey rows written (0 if none).
    """
    time_range = get_session_time_range(session_folder)
    if time_range is None:
        return 0

    session_start_ts, session_end_ts = time_range
    nurse_id = extract_nurse_id_from_folder(session_folder.name)

    # Surveys from this nurse whose interval is contained in this session
    matches = surveys_df[
        (surveys_df["nurse_id"] == nurse_id)
        & (surveys_df["survey_start_ts"] >= session_start_ts)
        & (surveys_df["survey_end_ts"] <= session_end_ts)
    ]

    if matches.empty:
        return 0

    # Convert to (start_sec, end_sec, label) relative to session start
    rows = []
    for _, row in matches.iterrows():
        start_sec = row["survey_start_ts"] - session_start_ts
        end_sec = row["survey_end_ts"] - session_start_ts
        rows.append((float(start_sec), float(end_sec), int(row["label"])))

    rows.sort(key=lambda r: r[0])

    output_path = output_dir / f"{session_folder.name}.csv"
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["start_sec", "end_sec", "label"])
        writer.writerows(rows)

    return len(rows)


def main():
    survey_xlsx_path = Path("data_extracted/nurse/SurveyResults.xlsx")
    nurse_data_dir = Path("data_extracted/nurse")
    output_dir = Path("labels/nurse")

    if not survey_xlsx_path.exists():
        raise FileNotFoundError(f"Survey file not found: {survey_xlsx_path}")
    if not nurse_data_dir.exists():
        raise FileNotFoundError(f"Nurse data folder not found: {nurse_data_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading surveys from {survey_xlsx_path}")
    surveys_df = parse_surveys(survey_xlsx_path)
    print(f"  {len(surveys_df)} usable surveys (binary label 0 or 1)")
    print(f"    non-stress (0): {(surveys_df['label'] == 0).sum()}")
    print(f"    stress (1):     {(surveys_df['label'] == 1).sum()}")

    session_folders = sorted(p for p in nurse_data_dir.iterdir() if p.is_dir())
    print(f"\nScanning {len(session_folders)} session folder(s)...")

    sessions_with_labels = 0
    total_rows_written = 0
    nurse_session_counts = {}

    for session_folder in session_folders:
        n = process_session(session_folder, surveys_df, output_dir)
        if n > 0:
            sessions_with_labels += 1
            total_rows_written += n
            nurse_id = extract_nurse_id_from_folder(session_folder.name)
            nurse_session_counts[nurse_id] = nurse_session_counts.get(nurse_id, 0) + n

    print(f"\nDone.")
    print(f"  Sessions with at least one labeled survey: {sessions_with_labels}/{len(session_folders)}")
    print(f"  Total labeled survey rows written:         {total_rows_written}")
    print(f"\nLabeled survey rows per nurse:")
    for nurse_id in sorted(nurse_session_counts):
        print(f"  {nurse_id}: {nurse_session_counts[nurse_id]} rows")


if __name__ == "__main__":
    main()