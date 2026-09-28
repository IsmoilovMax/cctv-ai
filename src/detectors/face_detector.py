import cv2


class FaceDetector:

    def __init__(self, model_path):
        self.detector = cv2.FaceDetectorYN.create(
            model_path, "", (320, 320), 0.9, 0.3, 5000
        )

    def detect(self, frame):
        height, width = frame.shape[:2]

        self.detector.setInputSize((width, height))

        _, faces = self.detector.detect(frame)

        return faces
