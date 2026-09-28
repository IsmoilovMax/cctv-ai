import cv2
from ultralytics import YOLO

# VIDEO_PATH = "videos/standing-fall.mp4"
VIDEO_PATH = "videos/face-demographics-walking.mp4"
model = YOLO("yolo26n-pose.pt")

printed = False

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Video could not be opened")


while True:

    ret, frame = cap.read()

    if not ret:
        print("Video finished")
        break

    results = model(frame, device="cpu", verbose=False)

    if not printed and results[0].keypoints is not None:

        print("Keypoints:")
        print(results[0].keypoints.xy)

        printed = True

    annotated_frame = results[0].plot()

    cv2.imshow("Pose Test", annotated_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
