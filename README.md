# Emotion Detection

A Windows desktop application that detects facial expressions in an uploaded
image. OpenCV's Haar Cascade locates faces, and a convolutional neural network
(CNN) classifies each detected face into one of seven emotion categories.

This repository includes the GUI inference application and a separate Jupyter
notebook for training and evaluating the CNN. Running the GUI uses the
checked-in model; it does not train the model.

## Emotion labels

The GUI's output labels, in the order expected by the model, are:

1. Angry
2. Disgust
3. Fear
4. Happy
5. Neutral
6. Sad
7. Surprise

The order is defined by `EMOTIONS_LIST` in `gui.py` and corresponds to the
model's seven softmax outputs.

## Project structure

```text
Emotion_detection-main/
├── .venv/                               # Local Python environment (not required in Git)
├── .vscode/                             # VS Code workspace settings
├── gui.py                              # Windows image-upload GUI and inference
├── model_creation.ipynb                # Separate model training/evaluation workflow
├── model_a1.json                       # Saved Keras CNN architecture
├── model_weights1.h5                   # Trained CNN weights used by the GUI
├── haarcascade_frontalface_default.xml  # OpenCV frontal-face detector
├── requirements.txt                    # Pinned Python dependencies
├── training_outputs/                    # Created when notebook training runs
└── README.md
```

The `.venv/` and `training_outputs/` directories are local/generated content;
the notebook creates timestamped training artifacts under `training_outputs/`.

## Model and face detector

- **CNN:** accepts a 48 x 48 grayscale face crop and returns seven class
  probabilities. The notebook's CNN definition has four convolutional blocks
  followed by dense layers and a seven-unit softmax output.
- **`model_a1.json`:** stores the Keras model architecture, not the learned
  parameter values.
- **`model_weights1.h5`:** stores the trained parameter values loaded by the
  GUI together with the JSON architecture.
- **`haarcascade_frontalface_default.xml`:** the OpenCV Haar Cascade classifier
  used to locate frontal faces before emotion classification. It is a face
  detector, not the emotion classifier.

For each detected face, the GUI converts the image to grayscale, crops the face,
resizes it to 48 x 48 pixels, and sends it to the CNN. The interface displays
the predicted label. If an image contains multiple faces, the current GUI
displays the label from the last face processed.

## Prerequisites

- Windows
- Python 3.11, including Tk support for the desktop GUI
- The Python launcher (`py`) and pip
- The project files listed above

The exact runtime dependencies are pinned in `requirements.txt`.

## Installation

Open PowerShell, change to the project directory, and create the project-local
virtual environment:

```powershell
Set-Location 'C:\Users\vipin\Downloads\Emotion_detection-main\Emotion_detection-main'
py -3.11 --version
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run these commands from the project directory. Use the `.venv` Python
executable for subsequent commands so the dependencies are installed and used
in the project environment, not globally.

## Run the desktop application

From the project directory, run:

```powershell
.\.venv\Scripts\python.exe gui.py
```

In the application, select **Upload Image**, choose an image containing a
visible frontal face, and click **Detect Emotion**. The GUI supports PNG, JPEG,
BMP, and other formats readable by both Pillow and OpenCV. It reports when an
image cannot be read or no face is detected.

The model and Haar Cascade are loaded relative to `gui.py`, so keep
`model_a1.json`, `model_weights1.h5`, and
`haarcascade_frontalface_default.xml` in the project directory alongside the
GUI script.

## Train and evaluate the model

Training is separate from running `gui.py`. The notebook constructs and trains
a CNN; running its training cells is not required for GUI inference. The
checked-in model files are used by the GUI, and notebook outputs are written to
a new timestamped directory under `training_outputs/` rather than replacing
those files.

### Dataset

The notebook requires a FER-style image-folder dataset with separate training
and test splits, each containing the same seven class folders:

```text
fer2013/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

Place supported image files inside the corresponding class directories. A
suitable source is the
[FER-2013 image-folder dataset on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013).
Download and extract the image-folder archive; a CSV-only dataset is not the
folder format this notebook expects.

The notebook selects the dataset root in this order:

1. The path in the `FER_DATASET_DIR` environment variable, if set.
2. `D:\datasets\fer2013`, if it contains both `train/` and `test/`.
3. `train/` and `test/` under the project directory.

For a dataset stored elsewhere, set the environment variable before starting
VS Code so the notebook kernel inherits it:

```powershell
$env:FER_DATASET_DIR = 'D:\datasets\fer2013'
code 'C:\Users\vipin\Downloads\Emotion_detection-main\Emotion_detection-main'
```

If VS Code is already open, close it before launching it from that PowerShell
session. In VS Code, select the project's `.venv` Python kernel for
`model_creation.ipynb`.

### Notebook checks and outputs

Before training, the notebook checks the split directories, supported and
readable images, matching class folders, and the seven expected labels. It
removes redundant byte-identical images within a split, keeps images shared
between splits only in training, and excludes image content assigned
conflicting labels. It then reserves a stratified 20% of the cleaned training
set for validation; the separate test split is held back for final evaluation.

Preprocessing uses OpenCV grayscale conversion and `INTER_LINEAR` resizing to
48 x 48 pixels. Pixel values remain in the 0–255 range to match the GUI
inference path. The notebook also checks that the defined CNN has compatible
input/output shapes and can load the checked-in GUI weights. That compatibility
check does not mean the notebook's new training model starts from those
weights.

Training is configured for 15 epochs. Evaluation metrics, plots, and saved
architecture/weights are written to the run-specific directory under
`training_outputs/`. Training results depend on running the notebook; no
accuracy or benchmark is claimed here.

## Troubleshooting

- **`ModuleNotFoundError` or missing dependency:** activate/use the project's
  `.venv` Python and install the pinned dependencies with
  `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`.
- **Tkinter import or window error:** confirm the Python 3.11 installation
  includes Tk support.
- **Model or cascade file not found:** keep the three inference assets beside
  `gui.py` and run the application from the project environment.
- **Notebook reports that the FER dataset is missing:** confirm the dataset
  root contains `train/` and `test/`, each with all seven class subfolders, and
  set `FER_DATASET_DIR` if it is not in one of the notebook's default locations.
- **Notebook reports missing/mismatched classes or unreadable images:** check
  the directory spelling, class names, and that the folders contain readable
  image files rather than only CSV metadata.
