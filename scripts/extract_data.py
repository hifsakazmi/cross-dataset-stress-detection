"""
Extract E4 CSV files from dataset zips into a clean flat structure.

Local:  python scripts/extract_data.py
Colab:  from scripts.extract_data import extract_wesad, extract_campanella, extract_nurse
"""

import zipfile
import io
import os
from pathlib import Path

E4_FILES = {"ACC.csv", "BVP.csv", "EDA.csv", "HR.csv", "IBI.csv", "TEMP.csv"}


def extract_wesad(drive_root, output_dir):
    """
    WESAD.zip → WESAD/S2/S2_E4_Data.zip → ACC.csv
    Skips .pkl files and RespiBAN data. Only extracts E4 CSVs.
    """
    zip_path = Path(drive_root) / "WESAD.zip"
    out_dir = Path(output_dir) / "wesad"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Extracting WESAD...")

    with zipfile.ZipFile(zip_path, "r") as outer_zip:
        # Find all E4 zip files inside (e.g., WESAD/S2/S2_E4_Data.zip)
        e4_zips = [n for n in outer_zip.namelist() if "E4" in n and n.endswith(".zip")]
        print(f"  Found {len(e4_zips)} subjects")

        for e4_zip_path in e4_zips:
            # Get subject ID: "WESAD/S2/S2_E4_Data.zip" → "S2"
            parts = Path(e4_zip_path).parts
            subject_id = parts[1]

            subject_out = out_dir / subject_id
            subject_out.mkdir(exist_ok=True)

            # Open the inner zip from memory
            e4_zip_bytes = outer_zip.read(e4_zip_path)
            with zipfile.ZipFile(io.BytesIO(e4_zip_bytes), "r") as inner_zip:
                for member in inner_zip.namelist():
                    filename = Path(member).name
                    if filename in E4_FILES:
                        with inner_zip.open(member) as src:
                            with open(subject_out / filename, "wb") as dst:
                                dst.write(src.read())

            extracted = [f.name for f in subject_out.glob("*.csv")]
            print(f"  {subject_id}: {extracted}")

    print("WESAD done.\n")


def extract_campanella(drive_root, output_dir):
    """
    Data_29_subjects.zip → Subjects/subject_01/ACC.csv
    """
    zip_path = Path(drive_root) / "Data_29_subjects.zip"
    out_dir = Path(output_dir) / "campanella"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Extracting Campanella...")

    with zipfile.ZipFile(zip_path, "r") as z:
        for name in z.namelist()[:30]:
            print(name)

    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.namelist():
            filename = Path(member).name
            # Match files like "Subjects/subject_01/ACC.csv"
            if filename in E4_FILES:
                # Get subject folder: "Subjects/subject_01/ACC.csv" → "subject_01"
                parts = Path(member).parts
                # Find the part that starts with "subject"
                subject_id = None
                for part in parts:
                    if part.lower().startswith("subject_"):
                        subject_id = part
                        break

                if subject_id is None:
                    continue

                subject_out = out_dir / subject_id
                subject_out.mkdir(exist_ok=True)

                with z.open(member) as src:
                    with open(subject_out / filename, "wb") as dst:
                        dst.write(src.read())

    # Print summary
    subjects = sorted([d.name for d in out_dir.iterdir() if d.is_dir()])
    for s in subjects:
        csvs = [f.name for f in (out_dir / s).glob("*.csv")]
        print(f"  {s}: {csvs}")

    print(f"Campanella done. {len(subjects)} subjects.\n")


def extract_nurse(drive_root, output_dir):
    """
    doi_...zip → Stress_dataset.zip → 5C/5C_1586886626.zip → ACC.csv
    Three levels of nesting. Takes the LAST zip per subject.
    """
    zip_path = Path(drive_root) / "doi_10_5061_dryad_5hqbzkh6f__v20210917.zip"
    out_dir = Path(output_dir) / "nurse"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Extracting Nurse dataset...")

    # Level 1: Open outer zip, find Stress_dataset.zip inside
    with zipfile.ZipFile(zip_path, "r") as outer_zip:
        stress_zip_name = None
        for name in outer_zip.namelist():
            if "Stress_dataset" in name and name.endswith(".zip"):
                stress_zip_name = name
                break

        if stress_zip_name is None:
            print("  ERROR: Could not find Stress_dataset.zip inside outer zip")
            return

        print(f"  Found: {stress_zip_name}")
        stress_zip_bytes = outer_zip.read(stress_zip_name)

    # Level 2: Open Stress_dataset.zip, find subject session zips
    with zipfile.ZipFile(io.BytesIO(stress_zip_bytes), "r") as stress_zip:
        all_names = stress_zip.namelist()

        # Find all session zips (e.g., 5C/5C_1586886626.zip)
        session_zips = [n for n in all_names if n.endswith(".zip")]

        # Group by subject: "5C/5C_1586886626.zip" → subject "5C"
        subjects = {}
        for sz in session_zips:
            parts = Path(sz).parts
            if len(parts) >= 2:
                subject_id = parts[0]
                if subject_id not in subjects:
                    subjects[subject_id] = []
                subjects[subject_id].append(sz)

        print(f"  Found {len(subjects)} subjects")

        # For each subject, take the LAST session zip
        for subject_id in sorted(subjects.keys()):
            session_list = sorted(subjects[subject_id])
            last_session = session_list[-1]
            session_name = Path(last_session).stem  # e.g., "5C_1586886626"

            subject_out = out_dir / f"{subject_id}_{session_name}"
            subject_out.mkdir(exist_ok=True)

            # Level 3: Open the session zip, extract CSVs
            session_bytes = stress_zip.read(last_session)
            with zipfile.ZipFile(io.BytesIO(session_bytes), "r") as session_zip:
                for member in session_zip.namelist():
                    filename = Path(member).name
                    if filename in E4_FILES:
                        with session_zip.open(member) as src:
                            with open(subject_out / filename, "wb") as dst:
                                dst.write(src.read())

            extracted = [f.name for f in subject_out.glob("*.csv")]
            print(f"  {subject_id} (session: {session_name}): {extracted}")

    print("Nurse done.\n")


if __name__ == "__main__":
    """
    For local testing. Update these paths to match your machine.
    """
    DRIVE_ROOT = "E:/Project/datasets"
    OUTPUT = "data_extracted"

    extract_campanella(DRIVE_ROOT, OUTPUT)
    #extract_wesad(DRIVE_ROOT, OUTPUT)
    #extract_nurse(DRIVE_ROOT, OUTPUT)