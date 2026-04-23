import cv2
import mediapipe as mp
import numpy as np
import pickle

# load trained model
with open("pose_model.pkl", "rb") as f:
    model = pickle.load(f)

# mediapipe setup
model_path = "pose_landmarker.task"
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE
)

# skeleton connections
CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
]

cap = cv2.VideoCapture(0)
print("Starting YoseLog... press Q to quit")

with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )

        result = landmarker.detect(mp_image)

        if result.pose_landmarks:
            h, w, _ = frame.shape
            landmarks = result.pose_landmarks[0]

            # draw skeleton
            for start, end in CONNECTIONS:
                x1 = int(landmarks[start].x * w)
                y1 = int(landmarks[start].y * h)
                x2 = int(landmarks[end].x * w)
                y2 = int(landmarks[end].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            for lm in landmarks:
                x = int(lm.x * w)
                y = int(lm.y * h)
                cv2.circle(frame, (x, y), 5, (255, 255, 255), -1)

            # extract keypoints for classifier
            keypoints = []
            for lm in landmarks:
                keypoints.append(lm.x)
                keypoints.append(lm.y)

            # predict pose
            keypoints = np.array(keypoints).reshape(1, -1)
            prediction = model.predict(keypoints)[0]
            confidence = model.predict_proba(keypoints).max() * 100

            # display prediction on screen
            cv2.putText(frame, f"{prediction} ({confidence:.0f}%)",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 255, 255), 3)

        cv2.imshow("YoseLog", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()