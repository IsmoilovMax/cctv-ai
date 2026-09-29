import cv2
import numpy as np
import openvino as ov
import time
from collections import deque

VIDEO_PATH = "videos/face-demographics-walking.mp4"

FACE_MODEL_PATH = "models/face_detection_yunet_2026may.onnx"

AGE_GENDER_MODEL_PATH = "models/genderage.onnx"

ANALYZE_INTERVAL = 1.0
TRACK_TIMEOUT = 10.0
IOU_THRESHOLD = 0.25
HISTORY_SIZE = 5


class Track:

    def __init__(self, track_id, bbox):

        self.id = track_id
        self.bbox = bbox

        self.last_seen = time.monotonic()
        self.last_analysis = 0

        self.age_history = deque(maxlen=HISTORY_SIZE)

        self.gender_history = deque(maxlen=HISTORY_SIZE)

        self.age = None
        self.gender = None


class Tracker:

    def __init__(self):

        self.tracks = {}
        self.next_id = 1

    def calculate_iou(self, box1, box2):

        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        xa = max(x1, x2)
        ya = max(y1, y2)

        xb = min(x1 + w1, x2 + w2)

        yb = min(y1 + h1, y2 + h2)

        intersection_w = max(0, xb - xa)

        intersection_h = max(0, yb - ya)

        intersection = intersection_w * intersection_h

        area1 = w1 * h1
        area2 = w2 * h2

        union = area1 + area2 - intersection

        if union <= 0:
            return 0

        return intersection / union

    def update(self, detections):

        now = time.monotonic()

        used_tracks = set()

        for bbox in detections:

            best_track = None
            best_iou = 0

            for track_id, track in self.tracks.items():

                if track_id in used_tracks:
                    continue

                iou = self.calculate_iou(bbox, track.bbox)

                if iou > best_iou:

                    best_iou = iou
                    best_track = track

            if best_track is not None and best_iou >= IOU_THRESHOLD:

                track = best_track

                track.bbox = bbox
                track.last_seen = now

                used_tracks.add(track.id)

            else:

                track = Track(self.next_id, bbox)

                self.tracks[self.next_id] = track

                used_tracks.add(self.next_id)

                self.next_id += 1

        old_tracks = []

        for track_id, track in self.tracks.items():

            if now - track.last_seen > TRACK_TIMEOUT:
                old_tracks.append(track_id)

        for track_id in old_tracks:

            del self.tracks[track_id]

        return self.tracks


def transform(image, center, output_size, scale):

    cx = center[0] * scale
    cy = center[1] * scale

    matrix = np.array(
        [
            [scale, 0, output_size / 2 - cx],
            [0, scale, output_size / 2 - cy],
        ],
        dtype=np.float32,
    )

    cropped = cv2.warpAffine(
        image,
        matrix,
        (output_size, output_size),
        borderValue=0.0,
    )

    return cropped


# Face detector

face_detector = cv2.FaceDetectorYN.create(
    FACE_MODEL_PATH,
    "",
    (320, 320),
    0.9,
    0.3,
    5000,
)


# Age / gender model

core = ov.Core()

model = core.read_model(AGE_GENDER_MODEL_PATH)

compiled_model = core.compile_model(model, "CPU")

input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)


# Tracker

tracker = Tracker()


# Video

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():

    raise RuntimeError("Video could not be opened")


while True:

    ret, frame = cap.read()

    if not ret:

        print("Video finished")
        break

    height, width = frame.shape[:2]

    face_detector.setInputSize((width, height))

    _, faces = face_detector.detect(frame)

    detections = []

    if faces is not None:

        for face in faces:

            x, y, w, h = face[:4]

            detections.append((float(x), float(y), float(w), float(h)))

    tracks = tracker.update(detections)

    now = time.monotonic()

    for track in tracks.values():

        x, y, w, h = track.bbox

        # ------------------------------------------
        # Analyze only once per second
        # ------------------------------------------

        if now - track.last_analysis >= ANALYZE_INTERVAL:

            center = (
                (x + x + w) / 2,
                (y + y + h) / 2,
            )

            scale = 96 / (max(w, h) * 1.5)

            face_img = transform(frame, center, 96, scale)

            input_image = face_img.astype(np.float32)

            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

            input_image = input_image.transpose(2, 0, 1)

            input_image = input_image[None, ...]

            results = compiled_model({input_layer: input_image})

            output = results[output_layer][0]

            gender_index = np.argmax(output[:2])

            age = float(output[2]) * 100

            gender = "Female" if gender_index == 0 else "Male"

            track.age_history.append(age)

            track.gender_history.append(gender)

            # --------------------------------------
            # Smooth age
            # --------------------------------------

            track.age = int(round(np.median(track.age_history)))

            # --------------------------------------
            # Smooth gender
            # --------------------------------------

            male_count = sum(gender == "Male" for gender in track.gender_history)

            female_count = sum(gender == "Female" for gender in track.gender_history)

            track.gender = "Male" if male_count >= female_count else "Female"

            track.last_analysis = now

        if now - track.last_seen > 0.2:
            continue

        # ------------------------------------------
        # Draw
        # ------------------------------------------

        x1 = max(0, int(x))

        y1 = max(0, int(y))

        x2 = min(width, int(x + w))

        y2 = min(height, int(y + h))

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if track.age is not None and track.gender is not None:

            label = f"ID {track.id} | " f"{track.gender} | " f"Age: {track.age}"

        else:

            label = f"ID {track.id} | " "Analyzing..."

        cv2.putText(
            frame,
            label,
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    cv2.imshow("GenderAge Tracking", frame)

    key = cv2.waitKey(33) & 0xFF

    if key == ord("q"):

        break


cap.release()
cv2.destroyAllWindows()
