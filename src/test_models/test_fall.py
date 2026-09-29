import cv2
from ultralytics import YOLO

from detectors.fall_detector import FallDetector

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

    # =====================================================
    # YOLO POSE
    # =====================================================

    results = model(
        frame,
        device="cpu",
        conf=0.5,
        verbose=False,
    )

    # =====================================================
    # DEFAULT STATE
    # =====================================================

    is_fall = False
    angle = 0
    ratio = 0

    # =====================================================
    # FALL DETECTION
    # =====================================================

    if results[0].keypoints is not None:

        keypoints = results[0].keypoints.xy

        if len(keypoints) > 0:

            person_keypoints = keypoints[0].cpu().numpy()

            (
                is_fall,
                angle,
                ratio,
            ) = fall_detector.detect(person_keypoints)

    # =====================================================
    # DRAW YOLO SKELETON
    # =====================================================

    annotated_frame = results[0].plot(labels=False)

    # =====================================================
    # FALL STATUS
    # =====================================================

    if is_fall:

        print(f"FALL DETECTED | " f"Angle: {angle:.1f} | " f"Ratio: {ratio:.2f}")

        status = "FALL DETECTED"
        color = (0, 0, 255)

    else:

        status = "NORMAL"
        color = (0, 255, 0)

    cv2.putText(
        annotated_frame,
        status,
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        color,
        3,
    )

    # =====================================================
    # SHOW
    # =====================================================

    cv2.imshow(
        "Fall Detection Test",
        annotated_frame,
    )

    key = cv2.waitKey(100) & 0xFF

    if key == ord("q"):
        break


cap.release()

cv2.destroyAllWindows()
