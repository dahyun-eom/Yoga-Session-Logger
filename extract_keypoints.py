import cv2
import mediapipe as mp
import numpy as np
import os
import csv

model_path = "pose_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE
)

POSES = ["warrior_1", "warrior_2", "tree", "downdog", "plank", "child"]

# CSV header — 33 keypoints x 2 (x, y) = 66 columns + label
header = []
for i in range(33):
    header.append(f"x{i}")
    header.append(f"y{i}")
header.append("label")

rows = []
success = 0
skipped = 0

with PoseLandmarker.create_from_options(options) as landmarker:
    for pose in POSES:
        folder = f"dataset/{pose}"
        images = os.listdir(folder)
        print(f"\nProcessing {pose} ({len(images)} images)...")

        for img_file in images:
            img_path = os.path.join(folder, img_file)

            # read image
            frame = cv2.imread(img_path)
            if frame is None:
                skipped += 1
                continue

            # convert to mediapipe image
            try:
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                )
                result = landmarker.detect(mp_image)
            except Exception as e:
                skipped += 1
                continue

            # skip if no landmarks detected
            if not result.pose_landmarks:
                skipped += 1
                continue

            # extract x, y for all 33 keypoints
            landmarks = result.pose_landmarks[0]
            row = []
            for lm in landmarks:
                row.append(round(lm.x, 4))
                row.append(round(lm.y, 4))
            row.append(pose)

            rows.append(row)
            success += 1

        print(f"  done — {success} total successful so far")

# save to CSV
with open("keypoints.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print(f"\nFinished!")
print(f"Saved: {success} samples")
print(f"Skipped: {skipped} images (no pose detected or bad image)")
print(f"CSV saved to keypoints.csv")