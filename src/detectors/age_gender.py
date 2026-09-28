import cv2
import numpy as np
import openvino as ov


class AgeGender:

    def __init__(self, model_path):

        core = ov.Core()

        model = core.read_model(model_path)

        self.compiled_model = core.compile_model(model, "CPU")

        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)

    def predict(self, face_img):

        input_image = cv2.resize(face_img, (96, 96))

        input_image = input_image.astype(np.float32)

        input_image = input_image.transpose(2, 0, 1)

        input_image = input_image[None, ...]

        results = self.compiled_model({self.input_layer: input_image})

        output = results[self.output_layer][0]

        female_probability = float(output[0])
        male_probability = float(output[1])

        age = float(output[2]) * 100

        return (age, female_probability, male_probability)
