from src.data_loader import load_subject
from src.data_loader import load_dataset
from src.labeling import get_campanella_labels
from src.labeling import get_wesad_labels
from src.labeling import get_nurse_labels
from src.windowing import create_windows
from pathlib import Path

def test_create_windows():
    # Step 1: Load signals
    signals, rates = load_subject(
        dataset_name="campanella",
        subject_id="subject_01",
        data_root="data_extracted"
    )
    if not signals:
        print(f"{signals}")

    # Step 2: Get total duration
    #duration = (signals["EDA"].index[-1] - signals["EDA"].index[0]).total_seconds()
    duration = min(
        (sig.index[-1] - sig.index[0]).total_seconds()
        for sig in signals.values()
        if len(sig) > 0
    )
    print (f"Recording duration: {duration:.0f} sec ({duration/60:.1f} min)")

    # Step 3: Generate phases from protocol
    phases = get_campanella_labels(duration)
    print (f"Phases: {phases}")

    # Step 4: Create windows using signals + phases
    windows = create_windows(signals, phases, total_duration=duration)

    # Check results
    stress_count = sum(1 for w in windows if w["label"] == 1)
    non_stress_count = sum(1 for w in windows if w["label"] == 0)

    print(f"Total windows: {len(windows)}")
    print(f"Stress: {stress_count}, Non-stress: {non_stress_count}")

def test_campanella_windowing():
    dataset = load_dataset("campanella", "data_extracted")

    for subject_id, data in dataset.items():
        try: 
            print(f"Testing windowing for {subject_id}")
            signals = data["signals"]
            duration = min(
                (sig.index[-1] - sig.index[0]).total_seconds()
                for sig in signals.values() if len(sig) > 0
            )
            phases = get_campanella_labels(duration)
            windows = create_windows(signals, phases)
            stress = sum(1 for w in windows if w["label"] == 1)
            non_stress = sum(1 for w in windows if w["label"] == 0)
            print(f"{subject_id}: {duration/60:.1f} min, {len(windows)} windows, stress={stress}, non-stress={non_stress}")
        except Exception as e:
            print(e)
            continue


def test_wesad_windowing():
    dataset = load_dataset("wesad", "data_extracted")

    for subject_id, data in dataset.items():
        try:
            print(f"Testing windowing for {subject_id}")
            signals = data["signals"]
            duration = min(
                (sig.index[-1] - sig.index[0]).total_seconds()
                for sig in signals.values() if len(sig) > 0
            )
            phases = get_wesad_labels(subject_id)
            windows = create_windows(signals, phases, total_duration=duration)
            stress = sum(1 for w in windows if w["label"] == 1)
            non_stress = sum(1 for w in windows if w["label"] == 0)
            print(f"{subject_id}: {duration/60:.1f} min, {len(windows)} windows, stress={stress}, non-stress={non_stress}")
        except Exception as e:
            print(e)
            continue

def test_nurse_windowing():
    dataset = load_dataset("nurse", "data_extracted")
    
    for session_id, data in dataset.items():
        try:
            signals = data["signals"]
            duration = min(
                (sig.index[-1] - sig.index[0]).total_seconds()
                for sig in signals.values() if len(sig) > 0
            )
            phases = get_nurse_labels(session_id)
            if not phases:
                continue  # no labeled surveys in this session
            windows = create_windows(signals, phases, total_duration=duration)
            stress = sum(1 for w in windows if w["label"] == 1)
            non_stress = sum(1 for w in windows if w["label"] == 0)
            nurse_id = session_id.split("_")[0]
            print(f"{session_id} (nurse {nurse_id}): {duration/60:.1f} min, {len(windows)} windows, stress={stress}, non-stress={non_stress}")
        except Exception as e:
            print(f"{session_id}: {e}")
            continue

if __name__ == "__main__": 
    #test_create_windows()
    #test_campanella_windowing()
    #test_wesad_windowing()
    test_nurse_windowing()