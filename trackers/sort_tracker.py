from trackers.sort import Sort
import numpy as np

class SimpleTracker:
    def __init__(self):
        self.tracker = Sort()

    def update(self, detections):
        # Detections are [x1, y1, x2, y2, conf, class_id]
        bboxes_only = [d[:4] for d in detections]  # Only first 4 values
        return self.tracker.update(np.array(bboxes_only))  # MUST convert to numpy array
