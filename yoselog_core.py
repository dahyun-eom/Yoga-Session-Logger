import csv
import os
import pickle
import threading
import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import cv2

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))

import mediapipe as mp
import numpy as np


MODEL_PATH = "pose_model.pkl"
LANDMARKER_PATH = "pose_landmarker.task"
LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
]

MIN_POSE_CONFIDENCE = 45.0
MIN_LANDMARK_VISIBILITY = 0.35
PAUSE_GESTURE_FRAMES = 18

ASYMMETRIC_POSES = {
    "warrior_1", "warrior_2", "warrior_3", "reverse_warrior",
    "triangle", "half_moon", "side_plank", "low_lunge",
    "extended_side", "gate", "reclining_toe", "tree", "eagle",
    "pigeon",
}


def ensure_landmarker_model(path=LANDMARKER_PATH):
    if os.path.exists(path):
        return path

    print("Downloading MediaPipe pose landmarker model...")
    urllib.request.urlretrieve(LANDMARKER_URL, path)
    return path


def landmarks_to_keypoints(landmarks):
    keypoints = []
    for landmark in landmarks:
        keypoints.append(landmark.x)
        keypoints.append(landmark.y)
    return np.array(keypoints)


def detect_side(landmarks, pose_name):
    return pose_name


def infer_side(landmarks, pose_name):
    if pose_name not in ASYMMETRIC_POSES or landmarks is None:
        return pose_name

    points = {
        index: np.array([landmarks[index].x, landmarks[index].y])
        for index in [11, 12, 15, 16, 23, 24, 25, 26, 27, 28]
    }

    def angle(a, b, c):
        ba = a - b
        bc = c - b
        denom = max(np.linalg.norm(ba) * np.linalg.norm(bc), 1e-6)
        cosine = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
        return np.degrees(np.arccos(cosine))

    left_knee_angle = angle(points[23], points[25], points[27])
    right_knee_angle = angle(points[24], points[26], points[28])
    left_ankle_y = points[27][1]
    right_ankle_y = points[28][1]
    left_wrist_y = points[15][1]
    right_wrist_y = points[16][1]
    left_shoulder_y = points[11][1]
    right_shoulder_y = points[12][1]

    if pose_name in {"warrior_1", "warrior_2", "low_lunge", "reverse_warrior"}:
        side = "left" if left_knee_angle < right_knee_angle else "right"
    elif pose_name in {"triangle", "extended_side", "half_moon", "gate"}:
        side = "left" if left_wrist_y > right_wrist_y else "right"
    elif pose_name == "side_plank":
        side = "left" if left_shoulder_y > right_shoulder_y else "right"
    elif pose_name in {"warrior_3", "reclining_toe", "tree", "eagle", "pigeon"}:
        side = "left" if left_ankle_y < right_ankle_y else "right"
    else:
        side = "left"

    return f"{pose_name}_{side}"


def calculate_balance_score(hold_keypoints):
    if len(hold_keypoints) < 5:
        return None

    kp_array = np.array(hold_keypoints)

    centers = []
    torso_lengths = []
    shoulder_tilts = []
    for row in kp_array:
        points = row.reshape(33, 2)
        left_shoulder = points[11]
        right_shoulder = points[12]
        left_hip = points[23]
        right_hip = points[24]

        shoulder_mid = (left_shoulder + right_shoulder) / 2
        hip_mid = (left_hip + right_hip) / 2
        centers.append((shoulder_mid + hip_mid) / 2)
        torso_lengths.append(max(np.linalg.norm(shoulder_mid - hip_mid), 0.05))
        shoulder_tilts.append(left_shoulder[1] - right_shoulder[1])

    centers = np.array(centers)
    center_sway = np.mean(np.linalg.norm(centers - np.mean(centers, axis=0), axis=1))
    normalized_sway = center_sway / np.mean(torso_lengths)
    tilt_wobble = float(np.std(shoulder_tilts))

    score = max(0, min(100, 100 - (normalized_sway * 180) - (tilt_wobble * 120)))
    return round(score, 1)


def keypoint_xy(keypoints, index):
    return keypoints[index * 2:index * 2 + 2]


def landmarks_are_visible(landmarks):
    required = [11, 12, 23, 24, 25, 26, 27, 28]
    for index in required:
        visibility = getattr(landmarks[index], "visibility", 1.0)
        presence = getattr(landmarks[index], "presence", 1.0)
        if visibility < MIN_LANDMARK_VISIBILITY or presence < MIN_LANDMARK_VISIBILITY:
            return False
    return True


def is_pause_gesture(landmarks):
    if not landmarks_are_visible(landmarks):
        return False

    left_shoulder = np.array([landmarks[11].x, landmarks[11].y])
    right_shoulder = np.array([landmarks[12].x, landmarks[12].y])
    left_wrist = np.array([landmarks[15].x, landmarks[15].y])
    right_wrist = np.array([landmarks[16].x, landmarks[16].y])
    nose_y = landmarks[0].y

    shoulder_width = max(np.linalg.norm(left_shoulder - right_shoulder), 0.05)

    hands_high = (
        left_wrist[1] < min(left_shoulder[1], nose_y + shoulder_width * 0.6)
        and right_wrist[1] < min(right_shoulder[1], nose_y + shoulder_width * 0.6)
    )
    hands_apart = abs(left_wrist[0] - right_wrist[0]) > shoulder_width * 0.55

    return hands_high and hands_apart


def is_active_pose_candidate(landmarks, confidence):
    if confidence < MIN_POSE_CONFIDENCE:
        return False
    if not landmarks_are_visible(landmarks):
        return False
    return True


def balance_label(score):
    if score is None:
        return "Collecting"
    if score >= 80:
        return "Excellent"
    if score >= 60:
        return "Steady"
    if score >= 40:
        return "Wobbly"
    return "Unstable"


def draw_skeleton(frame, landmarks):
    h, w, _ = frame.shape
    for start, end in CONNECTIONS:
        x1 = int(landmarks[start].x * w)
        y1 = int(landmarks[start].y * h)
        x2 = int(landmarks[end].x * w)
        y2 = int(landmarks[end].y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (47, 201, 142), 2)

    for landmark in landmarks:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)


@dataclass
class SessionState:
    active: bool = False
    paused: bool = False
    current_pose: str | None = None
    display_pose: str | None = None
    confidence: float = 0.0
    stable_frames: int = 0
    required_stable_frames: int = 30
    accumulated_time: float = 0.0
    balance: float | None = None
    balance_status: str = "Collecting"
    movement: float = 0.0
    session_log: list[dict] = field(default_factory=list)
    started_at: str | None = None
    message: str = "Ready"

    @property
    def stability_progress(self):
        if self.required_stable_frames <= 0:
            return 0.0
        return min(1.0, self.stable_frames / self.required_stable_frames)

    @property
    def total_time(self):
        return round(sum(entry["duration"] for entry in self.session_log), 1)

    @property
    def average_balance(self):
        scores = [entry["balance"] for entry in self.session_log if entry["balance"] is not None]
        if not scores:
            return None
        return round(sum(scores) / len(scores), 1)

    def to_dict(self):
        return {
            "active": self.active,
            "paused": self.paused,
            "current_pose": self.current_pose,
            "display_pose": self.display_pose,
            "confidence": round(self.confidence, 1),
            "stable_frames": self.stable_frames,
            "required_stable_frames": self.required_stable_frames,
            "stability_progress": round(self.stability_progress, 3),
            "accumulated_time": round(self.accumulated_time, 1),
            "balance": self.balance,
            "balance_status": self.balance_status,
            "movement": round(self.movement, 4),
            "session_log": list(self.session_log),
            "started_at": self.started_at,
            "total_time": self.total_time,
            "average_balance": self.average_balance,
            "message": self.message,
        }


class YogaSessionEngine:
    def __init__(self, camera_index=0, fps=30, stability_threshold=0.05, required_stable_sec=1.0):
        self.camera_index = camera_index
        self.fps = fps
        self.stability_threshold = stability_threshold
        self.required_stable_sec = required_stable_sec
        self.required_stable_frames = int(required_stable_sec * fps)

        self.lock = threading.Lock()
        self.state = SessionState(required_stable_frames=self.required_stable_frames)
        self.thread = None
        self.stop_event = threading.Event()
        self.latest_frame = None

        self.model = None
        self.prev_keypoints = None
        self.last_landmarks = None
        self.hold_keypoints = []
        self.keypoint_history = deque(maxlen=5)
        self.pause_gesture_frames = 0
        self.pause_gesture_armed = True

    def load_model(self):
        if self.model is None:
            with open(MODEL_PATH, "rb") as model_file:
                self.model = pickle.load(model_file)

    def start(self):
        with self.lock:
            if self.state.active:
                return
            self.state = SessionState(
                active=True,
                required_stable_frames=self.required_stable_frames,
                started_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                message="Starting camera",
            )
            self.prev_keypoints = None
            self.last_landmarks = None
            self.hold_keypoints = []
            self.keypoint_history.clear()
            self.pause_gesture_frames = 0
            self.pause_gesture_armed = True
            self.latest_frame = None

        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        self._log_current_pose()
        with self.lock:
            self.state.active = False
            self.state.message = "Session stopped"

    def reset(self):
        with self.lock:
            active = self.state.active
            self.state = SessionState(
                active=active,
                required_stable_frames=self.required_stable_frames,
                started_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S") if active else None,
                message="Session reset",
            )
            self.prev_keypoints = None
            self.last_landmarks = None
            self.hold_keypoints = []
            self.keypoint_history.clear()
            self.pause_gesture_frames = 0
            self.pause_gesture_armed = True

    def snapshot(self):
        with self.lock:
            return self.state.to_dict()

    def frame_bytes(self):
        with self.lock:
            if self.latest_frame is None:
                return None
            ok, buffer = cv2.imencode(".jpg", self.latest_frame)
        if not ok:
            return None
        return buffer.tobytes()

    def export_csv(self, path):
        with self.lock:
            rows = list(self.state.session_log)

        with open(path, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["pose", "duration", "balance", "logged_at"])
            writer.writeheader()
            writer.writerows(rows)

    def _run(self):
        try:
            self.load_model()
            ensure_landmarker_model()
        except Exception as exc:
            with self.lock:
                self.state.active = False
                self.state.message = f"Startup error: {exc}"
            return

        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=LANDMARKER_PATH),
            running_mode=VisionRunningMode.IMAGE,
        )

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            with self.lock:
                self.state.active = False
                self.state.message = "Camera could not be opened"
            return

        with PoseLandmarker.create_from_options(options) as landmarker:
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break

                annotated = self._process_frame(frame, landmarker)
                with self.lock:
                    self.latest_frame = annotated

        cap.release()

    def _process_frame(self, frame, landmarker):
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
        result = landmarker.detect(mp_image)
        annotated = frame.copy()

        if not result.pose_landmarks:
            self._update_message("Step into frame")
            return self._draw_overlay(annotated)

        landmarks = result.pose_landmarks[0]
        draw_skeleton(annotated, landmarks)
        keypoints = landmarks_to_keypoints(landmarks)
        self.keypoint_history.append(keypoints)
        smoothed = np.mean(self.keypoint_history, axis=0)

        prediction = self.model.predict(smoothed.reshape(1, -1))[0]
        confidence = self.model.predict_proba(smoothed.reshape(1, -1)).max() * 100

        self._update_session(smoothed, landmarks, prediction, confidence)
        return self._draw_overlay(annotated)

    def _update_session(self, keypoints, landmarks, prediction, confidence):
        with self.lock:
            self.state.confidence = confidence
            self.last_landmarks = landmarks

            if self.prev_keypoints is None or self.state.current_pose is None:
                self.state.current_pose = prediction
                self.state.display_pose = "holding..."
                self.prev_keypoints = keypoints
                self.state.message = "holding..."
                return

            diff = float(np.mean(np.abs(keypoints - self.prev_keypoints)))
            self.state.movement = diff

            if prediction == self.state.current_pose:
                if diff < self.stability_threshold:
                    self.state.stable_frames += 1
                    self.state.accumulated_time += 1.0 / self.fps
                    self.hold_keypoints.append(keypoints.copy())
                else:
                    self.state.stable_frames = 0
            else:
                self._log_current_pose_locked()
                self.state.current_pose = prediction
                self.state.stable_frames = 0
                self.state.accumulated_time = 0.0
                self.hold_keypoints = []

            if self.state.stable_frames >= self.required_stable_frames:
                self.state.display_pose = detect_side(landmarks, self.state.current_pose)
                self.state.message = f"{self.state.current_pose} ({confidence:.0f}%)"
            else:
                self.state.display_pose = "holding..."
                self.state.message = "holding..."

            self.state.balance = calculate_balance_score(self.hold_keypoints)
            self.state.balance_status = balance_label(self.state.balance)

            self.prev_keypoints = keypoints

    def _handle_pause_gesture(self, landmarks, keypoints):
        gesture_detected = is_pause_gesture(landmarks)
        with self.lock:
            if gesture_detected:
                self.pause_gesture_frames += 1
            else:
                self.pause_gesture_frames = 0
                self.pause_gesture_armed = True

            if self.pause_gesture_armed and self.pause_gesture_frames >= PAUSE_GESTURE_FRAMES:
                self.pause_gesture_armed = False
                if self.state.paused:
                    self.state.paused = False
                    self.state.current_pose = None
                    self.state.display_pose = "Ready"
                    self.state.stable_frames = 0
                    self.state.accumulated_time = 0.0
                    self.state.balance = None
                    self.state.balance_status = "Collecting"
                    self.state.message = "Logging resumed"
                    self.hold_keypoints = []
                    self.prev_keypoints = keypoints
                else:
                    self._log_current_pose_locked()
                    self.state.paused = True
                    self.state.current_pose = None
                    self.state.display_pose = "Paused"
                    self.state.stable_frames = 0
                    self.state.accumulated_time = 0.0
                    self.state.balance = None
                    self.state.balance_status = "Paused"
                    self.state.message = "Raise both hands again to resume"
                    self.hold_keypoints = []
                    self.prev_keypoints = keypoints

            if self.state.paused:
                self.state.display_pose = "Paused"
                self.state.confidence = 0.0
                self.state.message = "Raise both hands again to resume"
                return True

            if gesture_detected:
                self.state.display_pose = "Pause gesture"
                self.state.confidence = 0.0
                self.state.message = "Keep both hands raised"
                return True

        return False

    def _mark_no_active_pose(self, keypoints, confidence):
        with self.lock:
            self._log_current_pose_locked()
            self.state.current_pose = None
            self.state.display_pose = "No active pose"
            self.state.confidence = confidence
            self.state.stable_frames = 0
            self.state.accumulated_time = 0.0
            self.state.balance = None
            self.state.balance_status = "Collecting"
            self.state.movement = 0.0
            self.state.message = "Resting or low-confidence pose"
            self.hold_keypoints = []
            self.prev_keypoints = keypoints

    def _log_current_pose(self):
        with self.lock:
            self._log_current_pose_locked()

    def _log_current_pose_locked(self):
        if not self.state.current_pose or self.state.accumulated_time < self.required_stable_sec:
            return

        pose_name = detect_side(self.last_landmarks, self.state.current_pose)
        balance = calculate_balance_score(self.hold_keypoints)
        self.state.session_log.append({
            "pose": pose_name,
            "duration": round(self.state.accumulated_time, 1),
            "balance": balance,
            "logged_at": datetime.now().strftime("%H:%M:%S"),
        })

    def _update_message(self, message):
        with self.lock:
            self.state.message = message

    def _draw_overlay(self, frame):
        with self.lock:
            display_pose = self.state.display_pose or "No pose"
            paused = self.state.paused
            confidence = self.state.confidence
            progress = self.state.stability_progress
            hold_time = self.state.accumulated_time
            balance = self.state.balance
            message = self.state.message

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 104), (22, 30, 36), -1)
        title_color = (224, 168, 59) if paused else (255, 255, 255)
        if display_pose == "holding...":
            result_text = "holding..."
        else:
            result_text = f"{display_pose} ({confidence:.0f}%)"
        cv2.putText(frame, result_text, (18, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, title_color, 2)
        cv2.putText(frame, f"Holding: {hold_time:.1f}s", (18, 76),
            cv2.FONT_HERSHEY_SIMPLEX, 0.58, (198, 220, 210), 2)

        bar_width = int((frame.shape[1] - 36) * progress)
        cv2.rectangle(frame, (18, 88), (frame.shape[1] - 18, 96), (72, 84, 91), -1)
        cv2.rectangle(frame, (18, 88), (18 + bar_width, 96), (47, 201, 142), -1)

        if balance is not None:
            cv2.putText(frame, f"Balance {balance:.0f}%", (18, frame.shape[0] - 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (47, 201, 142), 2)

        return frame
