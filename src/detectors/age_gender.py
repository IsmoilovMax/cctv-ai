from collections import deque

import cv2
import numpy as np
import openvino as ov


class AgeGender:

    def __init__(
        self,
        model_path,
        history_size=5,
    ):
        core = ov.Core()

        model = core.read_model(model_path)

        self.compiled_model = core.compile_model(
            model,
            "CPU",
        )

        self.input_layer = self.compiled_model.input(0)

        self.age_output = self.compiled_model.output("fc1")

        self.history_size = history_size

        self.histories = {}

    def transform(
        self,
        image,
        center,
        output_size,
        scale,
    ):
        cx = center[0] * scale
        cy = center[1] * scale

        matrix = np.array(
            [
                [
                    scale,
                    0,
                    output_size / 2 - cx,
                ],
                [
                    0,
                    scale,
                    output_size / 2 - cy,
                ],
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

    def preprocess(self, frame, bbox):
        x, y, w, h = bbox

        center = (
            x + w / 2,
            y + h / 2,
        )

        face_size = max(w, h)

        scale = 96 / (face_size * 1.5)

        face_img = self.transform(
            frame,
            center,
            96,
            scale,
        )

        face_img = face_img.astype(np.float32)

        face_img = cv2.cvtColor(
            face_img,
            cv2.COLOR_BGR2RGB,
        )

        face_img = face_img.transpose(
            2,
            0,
            1,
        )

        face_img = face_img[None, ...]

        return face_img

    def predict(self, frame, bbox, track_id=None):
        input_image = self.preprocess(
            frame,
            bbox,
        )

        results = self.compiled_model({self.input_layer: input_image})

        output = results[self.age_output][0]

        gender_index = np.argmax(output[:2])

        age = float(output[2]) * 100

        if gender_index == 0:
            gender = "Female"
        else:
            gender = "Male"

        if track_id is None:
            return age, gender

        return self.smooth(
            track_id,
            age,
            gender,
        )

    def smooth(
        self,
        track_id,
        age,
        gender,
    ):
        if track_id not in self.histories:

            self.histories[track_id] = {
                "age": deque(maxlen=self.history_size),
                "gender": deque(maxlen=self.history_size),
            }

        history = self.histories[track_id]

        history["age"].append(age)

        history["gender"].append(gender)

        smoothed_age = float(np.median(history["age"]))

        gender_counts = {}

        for value in history["gender"]:

            gender_counts[value] = (
                gender_counts.get(
                    value,
                    0,
                )
                + 1
            )

        smoothed_gender = max(
            gender_counts,
            key=gender_counts.get,
        )

        return (
            smoothed_age,
            smoothed_gender,
        )

    def remove_track(
        self,
        track_id,
    ):
        self.histories.pop(
            track_id,
            None,
        )
