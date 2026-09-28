import math
from collections import deque


class FallDetector:

    def __init__(self, history_size=10, fall_angle=45, horizontal_ratio=1.2):

        self.history_size = history_size
        self.fall_angle = fall_angle
        self.horizontal_ratio = horizontal_ratio

        self.angle_history = deque(maxlen=history_size)

        self.is_fallen = False

    def calculate_body_angle(self, keypoints):

        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]

        left_hip = keypoints[11]
        right_hip = keypoints[12]

        shoulder_x = (left_shoulder[0] + right_shoulder[0]) / 2

        shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2

        hip_x = (left_hip[0] + right_hip[0]) / 2

        hip_y = (left_hip[1] + right_hip[1]) / 2

        dx = hip_x - shoulder_x
        dy = hip_y - shoulder_y

        angle = math.degrees(math.atan2(abs(dx), abs(dy)))

        return angle

    def calculate_body_ratio(self, keypoints):

        valid_points = []

        for point in keypoints:

            x, y = point

            if x > 0 and y > 0:

                valid_points.append((x, y))

        if not valid_points:
            return 0

        xs = [point[0] for point in valid_points]

        ys = [point[1] for point in valid_points]

        width = max(xs) - min(xs)
        height = max(ys) - min(ys)

        if height <= 0:
            return 0

        return width / height

    def detect(self, keypoints):

        angle = self.calculate_body_angle(keypoints)

        ratio = self.calculate_body_ratio(keypoints)

        self.angle_history.append(angle)

        recent_angles = list(self.angle_history)

        # Keep fall state once detected

        if self.is_fallen:

            return True, angle, ratio

        if len(recent_angles) < 3:

            return False, angle, ratio

        previous_angle = recent_angles[-3]

        angle_change = angle - previous_angle

        is_horizontal = angle >= self.fall_angle

        rapid_change = angle_change >= 20

        wide_body = ratio >= self.horizontal_ratio

        fall_detected = is_horizontal and (rapid_change or wide_body)

        if fall_detected:

            self.is_fallen = True

        return (self.is_fallen, angle, ratio)

    def reset(self):

        self.is_fallen = False
        self.angle_history.clear()
