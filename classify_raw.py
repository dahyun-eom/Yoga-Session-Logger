import cv2
import mediapipe as mp
import numpy as np
import pickle


MODEL_PATH = "pose_model.pkl"
LANDMARKER_PATH = "pose_landmarker.task"

CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
]


def keypoints_from_landmarks(landmarks):
    values = []
    for landmark in landmarks:
        values.append(landmark.x)
        values.append(landmark.y)
    return np.array(values)


def draw_skeleton(frame, landmarks):
    h, w, _ = frame.shape
    for start, end in CONNECTIONS:
        x1 = int(landmarks[start].x * w)
        y1 = int(landmarks[start].y * h)
        x2 = int(landmarks[end].x * w)
        y2 = int(landmarks[end].y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 220, 120), 2)

    for landmark in landmarks:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)


def main():
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=LANDMARKER_PATH),
        running_mode=VisionRunningMode.IMAGE,
    )

    cap = cv2.VideoCapture(0)
    print("Raw classifier test running. Press Q to quit.")

    with PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )
            result = landmarker.detect(mp_image)

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                draw_skeleton(frame, landmarks)

                keypoints = keypoints_from_landmarks(landmarks)
                probabilities = model.predict_proba(keypoints.reshape(1, -1))[0]
                classes = model.classes_
                top_indexes = np.argsort(probabilities)[::-1][:3]

                top_pose = classes[top_indexes[0]]
                top_confidence = probabilities[top_indexes[0]] * 100
                cv2.putText(
                    frame,
                    f"{top_pose} ({top_confidence:.1f}%)",
                    (20, 48),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),
                    2,
                )

                print(
                    " | ".join(
                        f"{classes[index]}: {probabilities[index] * 100:.1f}%"
                        for index in top_indexes
                    ),
                    end="\r",
                )
            else:
                cv2.putText(
                    frame,
                    "No body detected",
                    (20, 48),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),
                    2,
                )

            cv2.imshow("YoseLog Raw Classifier", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("\nDone.")


if __name__ == "__main__":
    main()
