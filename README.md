# YoseLog — Yoga Session Logger

Real-time yoga pose detection and session tracking using MediaPipe.

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
conda install -c conda-forge opencv numpy
pip install mediapipe
```

### Download pose model

The model file `pose_landmarker.task` will be downloaded automatically when you run the script for the first time.

## Run

```bash
conda activate yoselog
python test.py
```

Press **Q** to quit the camera window.

## Project Structure

YoseLog/
├── test.py # webcam + mediapipe skeleton test
├── pose_landmarker.task # mediapipe model (auto-downloaded)
└── README.md

## Tech Stack

- MediaPipe — body keypoint detection
- OpenCV — webcam capture and display
- Python 3.10
- NumPy — keypoint calculations
