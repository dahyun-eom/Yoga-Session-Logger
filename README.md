# YoseLog — Yoga Session Logger

Real-time yoga pose detection, balance scoring, and session tracking using MediaPipe and a trained Random Forest classifier.

## Dataset

This project uses the **Yoga-82** dataset for training the pose classifier.

> Verma, M., Kumawat, S., Nakashima, Y., & Raman, S. (2020). Yoga-82: A New Dataset for Fine-grained Classification of Human Poses. arXiv preprint arXiv:2004.10362.

Dataset: https://sites.google.com/view/yoga-82/home

### Supported Poses

- Warrior I
- Warrior II
- Tree Pose
- Downward Dog
- Plank
- Child Pose

## How the Classifier Works

1. Downloaded images from Yoga-82 dataset (~100 images per pose)
2. Extracted 33 body keypoints per image using MediaPipe Pose
3. Saved keypoints as a CSV file (66 features per sample)
4. Trained a Random Forest classifier (200 trees) using scikit-learn
5. **Accuracy: 83.5%** on held-out test set

## Setup

### Requirements

- Mac (Apple Silicon)
- Anaconda installed

### Create environment

```bash
conda create -n yoselog python=3.10
conda activate yoselog
```

### Install dependencies

```bash
conda install -c conda-forge opencv numpy pandas scikit-learn
/opt/anaconda3/envs/yoselog/bin/pip install mediapipe
```

### Model files

The following files are included in the repo and ready to use:

- `pose_model.pkl` — trained Random Forest classifier
- `keypoints.csv` — extracted keypoint dataset
- `pose_landmarker.task` — MediaPipe pose model (auto-downloaded on first run)

## Run

### Website

```bash
conda activate yoselog
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5001, then press **Start**.

To run the same app on a recorded video instead of the webcam:

```bash
YOSELOG_VIDEO="/path/to/session_video.mov" python app.py
```

The website shows:

- live webcam pose skeleton
- current pose and confidence
- hold timer and stability progress
- balance score based on pose sway
- automatic session log with duration and balance per pose
- CSV export for the session

The website currently logs the raw model pose label without adding left/right
side suffixes. This matches the older live-classifier behavior.

### OpenCV camera window

```bash
conda activate yoselog
python classify_live.py
```

Press **Q** to quit the camera window.

### Raw classifier test

Use this when you want to test only the trained model without website logging,
pause gestures, balance scoring, or stability rules:

```bash
python classify_raw.py
```

It shows the raw top prediction in the camera window and prints the top three
class probabilities in the terminal.

## Project Structure

YoseLog/

- classify_live.py # live pose classification
- classify_raw.py # raw model-only webcam test
- app.py # Flask website for live sessions
- yoselog_core.py # shared pose, stability, balance, and logging logic
- templates/index.html # session dashboard markup
- static/app.js # frontend state updates
- static/styles.css # dashboard styling
- extract_keypoints.py # extract keypoints from dataset images
- train_model.py # train Random Forest classifier
- download_images.py # download images from Yoga-82
- test.py # initial mediapipe skeleton test
- keypoints.csv # extracted keypoint data
- pose_model.pkl # trained classifier
- pose_landmarker.task # mediapipe model (auto-downloaded)
- README.md

## Tech Stack

- MediaPipe Pose — body keypoint detection (33 landmarks)
- OpenCV — webcam capture and display
- Python 3.10
- NumPy — keypoint calculations
- Pandas — data handling
- scikit-learn — Random Forest classifier
