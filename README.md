# Cross-dataset evaluation of wearable stress detection across three Empatica E4 datasets: from controlled lab to naturalistic clinical settings
Cross dataset stress detection 

## Datasets
|Dataset|Setting|Subjects|Stressor Type|Signals|
|-------|-------|--------|-------------|-------|
|WESAD|Lab (TSST protocol)|15|Psychological (Trier Social Stress Test)|E4 + RespiBAN chest|
|Campanella et al.|Lab (Lego/cognitive tasks)|29|Cognitive/task-based (Lego, math, etc.)|E4 only|
|Nurse dataset|Hospital (COVID ward)|15|Real-world occupational|E4 + RespiBAN chest|

## Setup
```console
git clone https://github.com/hifsakazm/cross-dataset-stress-detection.git
cd cross-dataset-stress-detection
```
### 1. Clone the repository
```console
# set DATASET = "dataset1" for 2-Class Dataset
DATASET = "dataset2" # This is 4-Class Dataset
```
### 2. Create virtual environment
```console
python -m venv stress-env
stress-env\Scripts\activate        # Windows
source stress-env/bin/activate     # Mac/Linux
```
### 3. Install dependencies
```console
pip install -r requirements.txt
```
### 4. Download datasets
See `data/README.md` for download links and folder placement.