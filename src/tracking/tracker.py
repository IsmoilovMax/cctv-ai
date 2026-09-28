import time
from collections import deque

class Track:

    def __init__(self, track_id, bbox):

        self.id = track_id
        self.bbox = bbox
        self.last_seen = time.monotonic()

        self.last_analysis = 0

        self.age_history = deque(maxlen=5)
        self.male_history = deque(maxlen=5)
        self.female_history = deque(maxlen=5)

        self.age = None
        self.gender = None


class Tracker:

    def __init__(self, iou_threshold=0.25, track_timeout=3.0):

        self.iou_threshold = iou_threshold
        self.track_timeout = track_timeout

        self.tracks = {}
        self.next_track_id = 1

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

            if best_track is not None and best_iou >= self.iou_threshold:

                track = best_track

                track.bbox = bbox
                track.last_seen = now

                used_tracks.add(track.id)

            else:

                track = Track(self.next_track_id, bbox)

                self.tracks[self.next_track_id] = track

                used_tracks.add(self.next_track_id)

                self.next_track_id += 1

        self.remove_old_tracks(now)

        return self.tracks

    def remove_old_tracks(self, now):

        old_tracks = []

        for track_id, track in self.tracks.items():

            if now - track.last_seen > self.track_timeout:

                old_tracks.append(track_id)

        for track_id in old_tracks:

            del self.tracks[track_id]
