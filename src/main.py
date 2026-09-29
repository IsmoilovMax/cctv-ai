import time

import cv2
from ultralytics import YOLO

from detectors.age_gender import AgeGender
from detectors.face_detector import FaceDetector
from detectors.fall_detector import FallDetector
from tracking.tracker import Tracker

# SETTINGS

FACE_MODEL = "models/face_detection_yunet_2026may.onnx"
AGE_GENDER_MODEL = "models/genderage.onnx"
FALL_MODEL = "yolo26n-pose.pt"
# VIDEO_PATH = "videos/face-demographics-walking.mp4"
VIDEO_PATH = "videos/standing-fall.mp4"

# Age/Gender modelni har necha sekundda ishlatish
ANALYZE_INTERVAL = 1.0

# Person ko'rinmay qolsa, trackni qancha vaqt saqlash
TRACK_TIMEOUT = 3.0

# Track matching
IOU_THRESHOLD = 0.25

# Fall detection confidence
FALL_CONFIDENCE = 0.5


# FACE DETECTOR

face_detector = FaceDetector(FACE_MODEL)


# AGE / GENDER

age_gender = AgeGender(AGE_GENDER_MODEL)


# FALL DETECTION

fall_model = YOLO(FALL_MODEL)

fall_detector = FallDetector()


# VIDEO

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Video could not be opened")


# TRACKING

tracker = Tracker(
    iou_threshold=IOU_THRESHOLD,
    track_timeout=TRACK_TIMEOUT,
)


# MAIN LOOP

while True:

    ret, frame = cap.read()

    if not ret:

        print("Video finished")

        break

    now = time.monotonic()

    height, width = frame.shape[:2]

    # FACE DETECTION

    faces = face_detector.detect(frame)

    detections = []

    if faces is not None:

        for face in faces:

            x, y, fw, fh = face[:4].astype(int)

            # Keep bbox inside frame

            x = max(0, x)

            y = max(0, y)

            fw = min(fw, width - x)

            fh = min(fh, height - y)

            if fw <= 0 or fh <= 0:
                continue

            detections.append(
                (
                    x,
                    y,
                    fw,
                    fh,
                )
            )

        # TRACKING

    tracks = tracker.update(detections)

    # AGE / GENDER ANALYSIS

    for track in tracks.values():

        if track.age is None or now - track.last_analysis >= ANALYZE_INTERVAL:

            try:

                age, gender = age_gender.predict(
                    frame,
                    track.bbox,
                    track.id,
                )

                track.age = age
                track.gender = gender

                track.last_analysis = now

            except Exception as e:

                print(
                    "Age/Gender error:",
                    e,
                )

        # FALL DETECTION

    is_fall = False
    fall_angle = 0
    fall_ratio = 0

    fall_results = fall_model(
        frame,
        device="cpu",
        conf=FALL_CONFIDENCE,
        verbose=False,
    )

    if fall_results[0].keypoints is not None:

        keypoints = fall_results[0].keypoints.xy

        if len(keypoints) > 0:

            person_keypoints = keypoints[0].cpu().numpy()

            (
                is_fall,
                fall_angle,
                fall_ratio,
            ) = fall_detector.detect(person_keypoints)

        # DRAW YOLO POSE

    frame = fall_results[0].plot(
        labels=False,
        boxes=False,
    )

    # DRAW FALL RESULT

    fall_label = "FALL DETECTED" if is_fall else "NORMAL"

    fall_color = (0, 0, 255) if is_fall else (0, 255, 0)

    cv2.putText(
        frame,
        fall_label,
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        fall_color,
        3,
    )

    # DRAW FACE / TRACK RESULTS

    for track in tracks.values():

        # Only draw currently visible tracks

        if now - track.last_seen > 0.15:
            continue

        x, y, fw, fh = track.bbox

        # Bounding box

        cv2.rectangle(
            frame,
            (x, y),
            (x + fw, y + fh),
            (0, 255, 0),
            2,
        )

        # Label

        if track.age is not None:

            label = f"ID: {track.id} | " f"{track.gender} | " f"Age: {track.age:.0f}"

        else:

            label = f"ID: {track.id} | " "Analyzing..."

        # Label background

        text_size = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            2,
        )[0]

        text_width = text_size[0]
        text_height = text_size[1]

        label_x1 = x

        label_y1 = max(
            0,
            y - text_height - 12,
        )

        label_x2 = x + text_width + 8

        label_y2 = y

        cv2.rectangle(
            frame,
            (label_x1, label_y1),
            (label_x2, label_y2),
            (0, 255, 0),
            -1,
        )

        # Label text

        cv2.putText(
            frame,
            label,
            (x + 4, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )

        # SHOW

    cv2.imshow(
        "CCTV AI",
        frame,
    )

    # KEYBOARD

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# CLEANUP

cap.release()
cv2.destroyAllWindows()
