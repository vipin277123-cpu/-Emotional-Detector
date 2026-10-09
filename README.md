# Emotion_detection

Simple emotion detection using a trained machine-learning model.

## Run on Windows

Use Python 3.11 and create a project-local virtual environment from this
directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe gui.py
```

The GUI lets you select an image and click **Detect Emotion**. The image should
contain a visible face. Model and Haar cascade files are loaded relative to the
project directory.

## Train the model

The training notebook expects the FER-2013 image-folder dataset. A suitable
source is the [FER-2013 Kaggle dataset](https://www.kaggle.com/datasets/msambare/fer2013).
Sign in to Kaggle, download the image-folder archive, and extract it so the
dataset root contains separate `train/` and `test/` folders in this layout:

```text
train/
  angry/
  disgust/
  fear/
  happy/
  neutral/
  sad/
  surprise/
test/
  angry/
  disgust/
  fear/
  happy/
  neutral/
  sad/
  surprise/
```

Put the corresponding images inside each class folder. The seven class folders
must match between `train/` and `test/`. The notebook skips unreadable images,
deduplicates identical files within each split, keeps shared images only in
`train/`, and excludes all copies of images assigned conflicting labels. It
reserves a stratified 20% of the cleaned training set for validation; `test/`
is used only for final evaluation. OpenCV performs the same grayscale
conversion, 48x48 resize, and unnormalized 0-255 preprocessing used by the
GUI. The notebook verifies the CNN accepts the checked-in GUI model's weights.
Training outputs, plots, and metrics go to a new timestamped folder under
`training_outputs/`; the existing `model_a1.json` and `model_weights1.h5` are
not overwritten. Select the **Python (.venv) - Emotion Detection** kernel for
`model_creation.ipynb` in VS Code.

The dataset may stay outside the project to avoid copying it. By default, the
notebook uses `D:\datasets\fer2013` when it contains both splits, and otherwise
checks for `train/` and `test/` under the project directory. To use a different
dataset root, set `FER_DATASET_DIR` before starting VS Code, for example:

```powershell
$env:FER_DATASET_DIR = 'D:\datasets\fer2013'
code 'C:\Users\vipin\Downloads\Emotion_detection-main\Emotion_detection-main'
```

If VS Code is already running, close it first so the notebook kernel inherits
the new environment variable. Use the Kaggle image-folder distribution, not a
CSV-only FER-2013 download.
