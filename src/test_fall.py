import cv2
from ultralytics import YOLO

from detectors.fall_detector import FallDetector

# VIDEO_PATH = "videos/standing-fall.mp4"
# VIDEO_PATH = "videos/face-demographics-walking.mp4"
# VIDEO_PATH = "videos/walk_P01_T01_video.mp4"
# VIDEO_PATH = "videos/normal-activity.mp4"
VIDEO_PATH = "videos/fall/fall_bwd_P01_T04_video.mp4"

model = YOLO("yolo26n-pose.pt")

fall_detector = FallDetector()

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Video could not be opened")


while True:

    ret, frame = cap.read()

    if not ret:

        print("Video finished")

        break

    results = model(frame, device="cpu", conf=0.5, verbose=False)

    if results[0].keypoints is not None:

        keypoints = results[0].keypoints.xy

        if len(keypoints) > 0:

            person_keypoints = keypoints[0].cpu().numpy()

            is_fall, angle, ratio = fall_detector.detect(person_keypoints)

            if is_fall:

                print(
                    f"FALL DETECTED | " f"Angle: {angle:.1f} | " f"Ratio: {ratio:.2f}"
                )

                cv2.putText(
                    frame,
                    "FALL DETECTED",
                    (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3,
                )

            else:

                cv2.putText(
                    frame,
                    "NORMAL",
                    (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 0),
                    3,
                )

    annotated_frame = results[0].plot()

    cv2.putText(
        annotated_frame,
        "FALL DETECTED" if is_fall else "NORMAL",
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 255) if is_fall else (0, 255, 0),
        3,
    )

    cv2.imshow("Fall Detection Test", annotated_frame)

    key = cv2.waitKey(100) & 0xFF

    if key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
