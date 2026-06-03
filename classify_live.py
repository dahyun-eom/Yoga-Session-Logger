# import cv2
# import mediapipe as mp
# import numpy as np
# import pickle
# import time

# # load trained model
# with open("pose_model.pkl", "rb") as f:
#     model = pickle.load(f)

# # mediapipe setup
# model_path = "pose_landmarker.task"
# BaseOptions = mp.tasks.BaseOptions
# PoseLandmarker = mp.tasks.vision.PoseLandmarker
# PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
# VisionRunningMode = mp.tasks.vision.RunningMode

# options = PoseLandmarkerOptions(
#     base_options=BaseOptions(model_asset_path=model_path),
#     running_mode=VisionRunningMode.IMAGE
# )

# # skeleton connections
# CONNECTIONS = [
#     (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
#     (11, 23), (12, 24), (23, 24),
#     (23, 25), (25, 27), (24, 26), (26, 28),
# ]

# # ── STABILITY SETTINGS ─────────
# STABILITY_THRESHOLD = 0.05 #how musch body movement is allowed 
# REQUIRED_STABLE_SEC = 1.0 #how long you need to hold the pose
# FPS = 30
# REQUIRED_STABLE_FRAMES = int(REQUIRED_STABLE_SEC * FPS) 

# # ── SESSION Log List ────
# session_log = []

# # ── STATE ──────────────
# prev_keypoints = None
# stable_frames = 0
# current_pose = None
# hold_start_time = None
# last_logged_pose = None
# accumulated_time = 0.0

# cap = cv2.VideoCapture(0) #realtime
# #cap = cv2.VideoCapture("dataset/test2_video.mov")
# print("Starting YoseLog... press Q to quit")

# with PoseLandmarker.create_from_options(options) as landmarker:
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         mp_image = mp.Image(
#             image_format=mp.ImageFormat.SRGB,
#             data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         )

#         result = landmarker.detect(mp_image)

#         if result.pose_landmarks:
#             h, w, _ = frame.shape
#             landmarks = result.pose_landmarks[0]

#             #draw skeleton
#             for start, end in CONNECTIONS:
#                 x1 = int(landmarks[start].x * w)
#                 y1 = int(landmarks[start].y * h)
#                 x2 = int(landmarks[end].x * w)
#                 y2 = int(landmarks[end].y * h)
#                 cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             for lm in landmarks:
#                 x = int(lm.x * w)
#                 y = int(lm.y * h)
#                 cv2.circle(frame, (x, y), 5, (255, 255, 255), -1)

#             #extract keypoints
#             keypoints = []
#             for lm in landmarks:
#                 keypoints.append(lm.x)
#                 keypoints.append(lm.y)
#             keypoints = np.array(keypoints)

#             #predict pose using the model
#             prediction = model.predict(keypoints.reshape(1, -1))[0]
#             confidence = model.predict_proba(keypoints.reshape(1, -1)).max() * 100

#             # ── STABILITY DETECTION ─────
#             if prev_keypoints is not None:
#                 diff = np.mean(np.abs(keypoints - prev_keypoints))

#                 if prediction == current_pose:
#                     if diff < STABILITY_THRESHOLD:
#                         # stable — accumulate time
#                         stable_frames += 1
#                         accumulated_time += 1.0 / FPS
#                     # if moved slightly but same pose — don't reset accumulated_time
#                     # just reset stable_frames so bar reacts
#                     else:
#                         stable_frames = 0

#                 else:
#                     # pose changed — log previous pose
#                     if current_pose is not None and accumulated_time >= REQUIRED_STABLE_SEC:
#                         session_log.append({
#                             "pose": current_pose,
#                             "duration": round(accumulated_time, 1)
#                         })
#                         print(f"Logged: {current_pose} — {accumulated_time:.1f}s")

#                     # reset for new pose
#                     stable_frames = 0
#                     accumulated_time = 0.0
#                     current_pose = prediction
#                     hold_start_time = time.time()
#                     last_logged_pose = current_pose

#             else:
#                 current_pose = prediction
#                 hold_start_time = time.time()

#             prev_keypoints = keypoints

#             # ── DISPLAY ──────────────────────────────────────────
#             # pose name and confidence
#             # cv2.putText(frame, f"{prediction} ({confidence:.0f}%)",
#             #     (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 255), 4)
#             #     # (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
#             if stable_frames >= REQUIRED_STABLE_FRAMES:
#                 cv2.putText(frame, f"{current_pose} ({confidence:.0f}%)",
#                     (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 4)
#             else:
#                 cv2.putText(frame, "holding...",
#                     (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

#             # live hold timer — shows accumulating seconds
#             cv2.putText(frame, f"Holding: {accumulated_time:.1f}s",
#                 (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

#             # stability progress bar
#             progress = min(stable_frames / REQUIRED_STABLE_FRAMES, 1.0)
#             bar_width = int(400 * progress)
#             cv2.rectangle(frame, (20, 100), (420, 120), (50, 50, 50), -1)
#             cv2.rectangle(frame, (20, 100), (20 + bar_width, 120), (0, 255, 0), -1)

#             # session log on screen
#             cv2.putText(frame, "Session Log:",
#                 (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
#             for i, entry in enumerate(session_log[-5:]):
#                 cv2.putText(frame,
#                     f"  {entry['pose']} — {entry['duration']}s",
#                     (20, 178 + i * 25),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)

#         cv2.imshow("YoseLog", frame)

#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

# #log the last pose when quitting
# if current_pose is not None and accumulated_time >= REQUIRED_STABLE_SEC:
#     session_log.append({
#         "pose": current_pose,
#         "duration": round(accumulated_time, 1)
#     })
#     print(f"Logged: {current_pose} — {accumulated_time:.1f}s")

# cap.release()
# cv2.destroyAllWindows()

# # ── FINAL SUMMARY ─────
# print("\n===== SESSION SUMMARY =====")
# if session_log:
#     for entry in session_log:
#         print(f"  {entry['pose']:15} — {entry['duration']}s")
#     total = sum(e['duration'] for e in session_log)
#     print(f"\n  Total session time: {total:.1f}s")
# else:
#     print("  No poses logged this session.")
# print("===========================")

# # # ── FINAL SUMMARY ────
# # print("\n===== SESSION SUMMARY =====")
# # if session_log:
# #     # group by pose and sum durations
# #     pose_totals = {}
# #     for entry in session_log:
# #         pose = entry["pose"]
# #         if pose not in pose_totals:
# #             pose_totals[pose] = 0.0
# #         pose_totals[pose] += entry["duration"]

# #     for pose, total_duration in pose_totals.items():
# #         print(f"  {pose:15} — {total_duration:.1f}s")

# #     total = sum(e['duration'] for e in session_log)
# #     print(f"\n  Total session time: {total:.1f}s")
# # else:
# #     print("  No poses logged this session.")
# # print("===========================")




########################version2 left/right 까지만 추가#####################3333333


# import cv2
# import mediapipe as mp
# import numpy as np
# import pickle
# import time
# from collections import deque

# # load trained model
# with open("pose_model.pkl", "rb") as f:
#     model = pickle.load(f)

# # mediapipe setup
# model_path = "pose_landmarker.task"
# BaseOptions = mp.tasks.BaseOptions
# PoseLandmarker = mp.tasks.vision.PoseLandmarker
# PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
# VisionRunningMode = mp.tasks.vision.RunningMode

# options = PoseLandmarkerOptions(
#     base_options=BaseOptions(model_asset_path=model_path),
#     running_mode=VisionRunningMode.IMAGE
# )

# # skeleton connections
# CONNECTIONS = [
#     (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
#     (11, 23), (12, 24), (23, 24),
#     (23, 25), (25, 27), (24, 26), (26, 28),
# ]

# # ── ASYMMETRIC POSES ───────────────────────────────────────
# ASYMMETRIC_POSES = [
#     "warrior_1", "warrior_2", "warrior_3",
#     "reverse_warrior", "triangle", "half_moon",
#     "side_plank", "low_lunge", "extended_side",
#     "gate", "reclining_toe", "tree",
#     "eagle", "pigeon"
# ]

# def detect_side(landmarks, pose_name):
#     if pose_name not in ASYMMETRIC_POSES:
#         return pose_name

#     left_hip_x       = landmarks[23].x
#     right_hip_x      = landmarks[24].x
#     left_wrist_y     = landmarks[15].y
#     right_wrist_y    = landmarks[16].y
#     left_ankle_y     = landmarks[27].y
#     right_ankle_y    = landmarks[28].y
#     left_shoulder_y  = landmarks[11].y
#     right_shoulder_y = landmarks[12].y

#     if pose_name in ["warrior_1", "warrior_2", "low_lunge", "reverse_warrior"]:
#         side = "left" if left_hip_x < right_hip_x else "right"

#     elif pose_name in ["triangle", "extended_side", "half_moon", "gate"]:
#         side = "left" if left_wrist_y > right_wrist_y else "right"

#     elif pose_name == "side_plank":
#         side = "left" if left_shoulder_y < right_shoulder_y else "right"

#     elif pose_name in ["warrior_3", "reclining_toe", "tree", "eagle", "pigeon"]:
#         side = "left" if left_ankle_y < right_ankle_y else "right"

#     else:
#         side = "left"

#     return f"{pose_name}_{side}"

# # ── STABILITY SETTINGS ─────────
# STABILITY_THRESHOLD = 0.05
# REQUIRED_STABLE_SEC = 1.0
# FPS = 30
# REQUIRED_STABLE_FRAMES = int(REQUIRED_STABLE_SEC * FPS)

# # ── SESSION LOG ────
# session_log = []

# # ── STATE ──────────────
# prev_keypoints = None
# stable_frames = 0
# current_pose = None
# hold_start_time = None
# last_logged_pose = None
# accumulated_time = 0.0
# last_landmarks = None

# # ── SMOOTHING ──────────
# keypoint_history = deque(maxlen=5)

# cap = cv2.VideoCapture(0)
# print("Starting YoseLog... press Q to quit")

# with PoseLandmarker.create_from_options(options) as landmarker:
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         mp_image = mp.Image(
#             image_format=mp.ImageFormat.SRGB,
#             data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         )

#         result = landmarker.detect(mp_image)

#         if result.pose_landmarks:
#             h, w, _ = frame.shape
#             landmarks = result.pose_landmarks[0]

#             # draw skeleton
#             for start, end in CONNECTIONS:
#                 x1 = int(landmarks[start].x * w)
#                 y1 = int(landmarks[start].y * h)
#                 x2 = int(landmarks[end].x * w)
#                 y2 = int(landmarks[end].y * h)
#                 cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             for lm in landmarks:
#                 x = int(lm.x * w)
#                 y = int(lm.y * h)
#                 cv2.circle(frame, (x, y), 5, (255, 255, 255), -1)

#             # extract keypoints
#             keypoints = []
#             for lm in landmarks:
#                 keypoints.append(lm.x)
#                 keypoints.append(lm.y)
#             keypoints = np.array(keypoints)

#             # smooth keypoints over last 5 frames
#             keypoint_history.append(keypoints)
#             keypoints = np.mean(keypoint_history, axis=0)

#             # predict pose
#             prediction = model.predict(keypoints.reshape(1, -1))[0]
#             confidence = model.predict_proba(keypoints.reshape(1, -1)).max() * 100

#             # ── STABILITY DETECTION ─────
#             if prev_keypoints is not None:
#                 diff = np.mean(np.abs(keypoints - prev_keypoints))

#                 if prediction == current_pose:
#                     if diff < STABILITY_THRESHOLD:
#                         stable_frames += 1
#                         accumulated_time += 1.0 / FPS
#                     else:
#                         stable_frames = 0
#                 else:
#                     if current_pose is not None and accumulated_time >= REQUIRED_STABLE_SEC:
#                         # session_log.append({
#                         #     "pose": current_pose,
#                         #     "duration": round(accumulated_time, 1)
#                         # })
#                         pose_with_side = detect_side(landmarks, current_pose)
#                         session_log.append({
#                             "pose": pose_with_side,
#                             "duration": round(accumulated_time, 1)
#                         })
#                         print(f"Logged: {current_pose} — {accumulated_time:.1f}s")

#                     stable_frames = 0
#                     accumulated_time = 0.0
#                     current_pose = prediction
#                     hold_start_time = time.time()
#                     last_logged_pose = current_pose

#             else:
#                 current_pose = prediction
#                 hold_start_time = time.time()

#             last_landmarks = landmarks

#             prev_keypoints = keypoints

#             # ── DISPLAY ──────────────
#             # pose name — big, only when confirmed

#             # if stable_frames >= REQUIRED_STABLE_FRAMES:
#             #     cv2.putText(frame, f"{current_pose} ({confidence:.0f}%)",
#             #         (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 4)
#             if stable_frames >= REQUIRED_STABLE_FRAMES:
#                 pose_with_side = detect_side(landmarks, current_pose)
#                 cv2.putText(frame, f"{pose_with_side} ({confidence:.0f}%)",
#                     (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)
            
#             else:
#                 cv2.putText(frame, "holding...",
#                     (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

#             # hold timer
#             cv2.putText(frame, f"Holding: {accumulated_time:.1f}s",
#                 (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

#             # progress bar
#             progress = min(stable_frames / REQUIRED_STABLE_FRAMES, 1.0)
#             bar_width = int(400 * progress)
#             cv2.rectangle(frame, (20, 145), (420, 165), (50, 50, 50), -1)
#             cv2.rectangle(frame, (20, 145), (20 + bar_width, 165), (0, 255, 0), -1)

#             # session log
#             cv2.putText(frame, "Session Log:",
#                 (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
#             for i, entry in enumerate(session_log[-5:]):
#                 cv2.putText(frame,
#                     f"  {entry['pose']} — {entry['duration']}s",
#                     (20, 225 + i * 25),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)

#         cv2.imshow("YoseLog", frame)

#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

# # log last pose
# # if current_pose is not None and accumulated_time >= REQUIRED_STABLE_SEC:
# #     session_log.append({
# #         "pose": current_pose,
# #         "duration": round(accumulated_time, 1)
# #     })
# if current_pose is not None and accumulated_time >= REQUIRED_STABLE_SEC:
#     pose_with_side = detect_side(last_landmarks, current_pose)
#     session_log.append({
#         "pose": pose_with_side,
#         "duration": round(accumulated_time, 1)
#     })
#     print(f"Logged: {current_pose} — {accumulated_time:.1f}s")

# cap.release()
# cv2.destroyAllWindows()

# # ── FINAL SUMMARY ─────
# print("\n===== SESSION SUMMARY =====")
# if session_log:
#     for entry in session_log:
#         print(f"  {entry['pose']:15} — {entry['duration']}s")
#     total = sum(e['duration'] for e in session_log)
#     print(f"\n  Total session time: {total:.1f}s")
# else:
#     print("  No poses logged this session.")
# print("===========================")




# #############################ver3 balance score 까지 추가################333

import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
from collections import deque

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

# ── ASYMMETRIC POSES ───────────────────────────────────────
ASYMMETRIC_POSES = [
    "warrior_1", "warrior_2", "warrior_3",
    "reverse_warrior", "triangle", "half_moon",
    "side_plank", "low_lunge", "extended_side",
    "gate", "reclining_toe", "tree",
    "eagle", "pigeon"
]

def detect_side(landmarks, pose_name):
    if pose_name not in ASYMMETRIC_POSES:
        return pose_name

    left_hip_x       = landmarks[23].x
    right_hip_x      = landmarks[24].x
    left_wrist_y     = landmarks[15].y
    right_wrist_y    = landmarks[16].y
    left_ankle_y     = landmarks[27].y
    right_ankle_y    = landmarks[28].y
    left_shoulder_y  = landmarks[11].y
    right_shoulder_y = landmarks[12].y

    if pose_name in ["warrior_1", "warrior_2", "low_lunge", "reverse_warrior"]:
        side = "left" if left_hip_x < right_hip_x else "right"
    elif pose_name in ["triangle", "extended_side", "half_moon", "gate"]:
        side = "left" if left_wrist_y > right_wrist_y else "right"
    elif pose_name == "side_plank":
        side = "left" if left_shoulder_y < right_shoulder_y else "right"
    elif pose_name in ["warrior_3", "reclining_toe", "tree", "eagle", "pigeon"]:
        side = "left" if left_ankle_y < right_ankle_y else "right"
    else:
        side = "left"

    return f"{pose_name}_{side}"

# ── BALANCE SCORE ──────────────────────────────────────────
def calculate_balance_score(hold_keypoints):
    if len(hold_keypoints) < 5:
        return None
    kp_array = np.array(hold_keypoints)
    variance = np.mean(np.var(kp_array, axis=0))
    score = max(0, 100 - (variance * 10000))
    return round(score, 1)

# ── STABILITY SETTINGS ─────────
STABILITY_THRESHOLD = 0.05
REQUIRED_STABLE_SEC = 1.0
FPS = 30
REQUIRED_STABLE_FRAMES = int(REQUIRED_STABLE_SEC * FPS)

# ── SESSION LOG ────
session_log = []

# ── STATE ──────────────
prev_keypoints = None
stable_frames = 0
current_pose = None
hold_start_time = None
last_logged_pose = None
accumulated_time = 0.0
last_landmarks = None
hold_keypoints = []

# ── SMOOTHING ──────────
keypoint_history = deque(maxlen=5)

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

            # extract keypoints
            keypoints = []
            for lm in landmarks:
                keypoints.append(lm.x)
                keypoints.append(lm.y)
            keypoints = np.array(keypoints)

            # smooth keypoints over last 5 frames
            keypoint_history.append(keypoints)
            keypoints = np.mean(keypoint_history, axis=0)

            # predict pose
            prediction = model.predict(keypoints.reshape(1, -1))[0]
            confidence = model.predict_proba(keypoints.reshape(1, -1)).max() * 100

            # ── STABILITY DETECTION ─────
            if prev_keypoints is not None:
                diff = np.mean(np.abs(keypoints - prev_keypoints))

                if prediction == current_pose:
                    if diff < STABILITY_THRESHOLD:
                        stable_frames += 1
                        accumulated_time += 1.0 / FPS
                        hold_keypoints.append(keypoints.copy())
                    else:
                        stable_frames = 0
                else:
                    if current_pose is not None and accumulated_time >= REQUIRED_STABLE_SEC:
                        pose_with_side = detect_side(landmarks, current_pose)
                        balance_score = calculate_balance_score(hold_keypoints)
                        session_log.append({
                            "pose": pose_with_side,
                            "duration": round(accumulated_time, 1),
                            "balance": balance_score
                        })
                        print(f"Logged: {pose_with_side} — {accumulated_time:.1f}s — balance: {balance_score}%")

                    stable_frames = 0
                    accumulated_time = 0.0
                    current_pose = prediction
                    hold_start_time = time.time()
                    last_logged_pose = current_pose
                    hold_keypoints = []

            else:
                current_pose = prediction
                hold_start_time = time.time()

            last_landmarks = landmarks
            prev_keypoints = keypoints

            # ── DISPLAY ──────────────
            if stable_frames >= REQUIRED_STABLE_FRAMES:
                pose_with_side = detect_side(landmarks, current_pose)
                cv2.putText(frame, f"{pose_with_side} ({confidence:.0f}%)",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)
            else:
                cv2.putText(frame, "holding...",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

            # hold timer
            cv2.putText(frame, f"Holding: {accumulated_time:.1f}s",
                (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # balance score — show live during hold
            if stable_frames >= REQUIRED_STABLE_FRAMES and hold_keypoints:
                balance = calculate_balance_score(hold_keypoints)
                if balance is not None:
                    color = (0, 255, 0) if balance >= 70 else (0, 165, 255) if balance >= 40 else (0, 0, 255)
                    cv2.putText(frame, f"Balance: {balance:.0f}%",
                        (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # progress bar
            progress = min(stable_frames / REQUIRED_STABLE_FRAMES, 1.0)
            bar_width = int(400 * progress)
            cv2.rectangle(frame, (20, 190), (420, 210), (50, 50, 50), -1)
            cv2.rectangle(frame, (20, 190), (20 + bar_width, 210), (0, 255, 0), -1)

            # session log
            cv2.putText(frame, "Session Log:",
                (20, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            for i, entry in enumerate(session_log[-5:]):
                balance_str = f" bal:{entry['balance']}%" if entry['balance'] else ""
                cv2.putText(frame,
                    f"  {entry['pose']} — {entry['duration']}s{balance_str}",
                    (20, 270 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)

        cv2.imshow("YoseLog", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# log last pose
if current_pose is not None and accumulated_time >= REQUIRED_STABLE_SEC:
    pose_with_side = detect_side(last_landmarks, current_pose)
    balance_score = calculate_balance_score(hold_keypoints)
    session_log.append({
        "pose": pose_with_side,
        "duration": round(accumulated_time, 1),
        "balance": balance_score
    })
    print(f"Logged: {pose_with_side} — {accumulated_time:.1f}s — balance: {balance_score}%")

cap.release()
cv2.destroyAllWindows()

# ── FINAL SUMMARY ─────
print("\n===== SESSION SUMMARY =====")
if session_log:
    for entry in session_log:
        balance_str = f"  balance: {entry['balance']}%" if entry['balance'] else ""
        print(f"  {entry['pose']:25} — {entry['duration']}s{balance_str}")
    total = sum(e['duration'] for e in session_log)
    print(f"\n  Total session time: {total:.1f}s")
else:
    print("  No poses logged this session.")
print("===========================")