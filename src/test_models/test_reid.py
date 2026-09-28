import cv2
import numpy as np
import openvino as ov

FACE_MODEL_PATH = "models/face_detection_yunet_2026may.onnx"
REID_MODEL_PATH = "models/w600k_mbf.onnx"
VIDEO_PATH = "videos/face-demographics-walking.mp4"


# -------------------------
# Face detector
# -------------------------

face_detector = cv2.FaceDetectorYN.create(
    FACE_MODEL_PATH,
    "",
    (320, 320),
    0.5,
    0.3,
    5000,
)


# -------------------------
# ReID model
# -------------------------

core = ov.Core()

model = core.read_model(REID_MODEL_PATH)

compiled_model = core.compile_model(
    model,
    "CPU",
)

input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)


# -------------------------
# ArcFace alignment
# -------------------------

ARC_FACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def align_face(frame, landmarks):

    transform = cv2.estimateAffinePartial2D(
        landmarks,
        ARC_FACE_TEMPLATE,
        method=cv2.LMEDS,
    )[0]

    if transform is None:
        return None

    aligned = cv2.warpAffine(
        frame,
        transform,
        (112, 112),
        borderValue=0,
    )

    return aligned


def cosine_similarity(embedding1, embedding2):

    return float(
        np.dot(
            embedding1,
            embedding2,
        )
    )


# -------------------------
# Generate embedding
# -------------------------


def get_embedding(face_image):

    face_image = cv2.cvtColor(
        face_image,
        cv2.COLOR_BGR2RGB,
    )

    face_image = face_image.astype(np.float32)

    # InsightFace ArcFace preprocessing
    face_image = (face_image - 127.5) / 127.5

    face_image = face_image.transpose(
        2,
        0,
        1,
    )

    face_image = face_image[None, ...]

    result = compiled_model({input_layer: face_image})

    embedding = result[output_layer][0]

    embedding = embedding / np.linalg.norm(embedding)

    return embedding


# -------------------------
# Video
# -------------------------

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():

    print("Failed to open video")

    exit()


frame_count = 0
embedding_count = 0
reference_embedding = None


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    height, width = frame.shape[:2]

    face_detector.setInputSize((width, height))

    _, faces = face_detector.detect(frame)

    if faces is not None:

        print(f"Frame {frame_count}: " f"Faces = {len(faces)}")

        for face in faces:

            # YuNet format:
            # x, y, w, h,
            # right_eye,
            # left_eye,
            # nose,
            # right_mouth,
            # left_mouth

            landmarks = np.array(
                [
                    face[4:6],
                    face[6:8],
                    face[8:10],
                    face[10:12],
                    face[12:14],
                ],
                dtype=np.float32,
            )

            aligned_face = align_face(
                frame,
                landmarks,
            )

            if aligned_face is None:
                continue

            embedding = get_embedding(aligned_face)

            if reference_embedding is None:

                reference_embedding = embedding

                print("Reference embedding saved")

            else:

                similarity = cosine_similarity(
                    reference_embedding,
                    embedding,
                )

                print("Similarity:", round(similarity, 4))

            embedding_count += 1

            print(
                "  Embedding:",
                embedding.shape,
                "Norm:",
                np.linalg.norm(embedding),
            )

            # Draw face box

            x, y, w, h = face[:4]

            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2,
            )

    cv2.imshow(
        "ReID Test",
        frame,
    )

    key = cv2.waitKey(30)

    if key == 27:
        break


cap.release()

cv2.destroyAllWindows()

print()
print(
    "Frames:",
    frame_count,
)

print(
    "Embeddings:",
    embedding_count,
)
