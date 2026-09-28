import cv2
import openvino as ov
import time
from collections import deque
from detectors.age_gender import AgeGender
from tracking.tracker import Tracker

# SETTINGS

FACE_MODEL = "models/face_detection_yunet_2026may.onnx"
# AGE_GENDER_MODEL = "models/age-gender-recognition-retail-0013.xml"
AGE_GENDER_MODEL = "models/genderage.onnx"
VIDEO_PATH = "videos/face-demographics-walking.mp4"
# VIDEO_PATH = "videos/fall/fall_bwd_P01_T03_video.mp4"

# Age/Gender modelni har necha sekundda ishlatish
ANALYZE_INTERVAL = 1.0

# Person ko'rinmay qolsa, trackni qancha vaqt saqlash
TRACK_TIMEOUT = 3.0

# Age history
AGE_HISTORY_SIZE = 5

# Track matching
IOU_THRESHOLD = 0.25

# FACE DETECTOR

from detectors.face_detector import FaceDetector

face_detector = FaceDetector(FACE_MODEL)

# OPENVINO AGE / GENDER

age_gender = AgeGender(AGE_GENDER_MODEL)

# =========================================================
# VIDEO
# =========================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Video could not be opened")


# =========================================================
# TRACKING
# =========================================================

tracker = Tracker(iou_threshold=IOU_THRESHOLD, track_timeout=TRACK_TIMEOUT)


# =========================================================
# MAIN LOOP
# =========================================================

prev_time = time.perf_counter()

while True:

    ret, frame = cap.read()

    if not ret:

        print("Video finished")

        break

    now = time.monotonic()

    h, w = frame.shape[:2]

    # =====================================================
    # FACE DETECTION
    # =====================================================

    faces = face_detector.detect(frame)

    detections = []

    if faces is not None:

        for face in faces:

            x, y, fw, fh = face[:4].astype(int)

            x = max(0, x)
            y = max(0, y)

            fw = min(fw, w - x)

            fh = min(fh, h - y)

            if fw <= 0 or fh <= 0:
                continue

            detections.append((x, y, fw, fh))

    # TRACKING
    tracks = tracker.update(detections)

    # AGE / GENDER ANALYSIS
    for track in tracks.values():

        if track.age is None or now - track.last_analysis >= ANALYZE_INTERVAL:

            x, y, fw, fh = track.bbox

            face_img = frame[y : y + fh, x : x + fw]

            if face_img.size > 0:

                try:

                    age, female_probability, male_probability = age_gender.predict(
                        face_img
                    )

                    track.age_history.append(age)
                    track.female_history.append(female_probability)
                    track.male_history.append(male_probability)

                    # Stable age

                    sorted_ages = sorted(track.age_history)

                    middle = len(sorted_ages) // 2

                    if len(sorted_ages) % 2:

                        track.age = sorted_ages[middle]

                    else:

                        track.age = (sorted_ages[middle - 1] + sorted_ages[middle]) / 2

                    # Stable gender

                    avg_female = sum(track.female_history) / len(track.female_history)

                    avg_male = sum(track.male_history) / len(track.male_history)

                    if avg_male > avg_female:
                        track.gender = "Male"
                    else:
                        track.gender = "Female"

                    track.last_analysis = now

                except Exception as e:

                    print("Age/Gender error:", e)

        # DRAW RESULTS

    for track in tracks.values():

        if now - track.last_seen > 0.15:
            continue

        x, y, fw, fh = track.bbox

        cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)

        if track.age is not None:

            label = f"ID: {track.id} | " f"{track.gender} | " f"Age: {track.age:.0f}"

        else:

            label = f"ID: {track.id} | Analyzing..."

        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

        cv2.rectangle(
            frame,
            (x, max(0, y - text_size[1] - 12)),
            (x + text_size[0] + 8, y),
            (0, 255, 0),
            -1,
        )

        cv2.putText(
            frame, label, (x + 4, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
        )

    # SHOW
    cv2.imshow("CCTV AI", frame)

    key = cv2.waitKey(33) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
