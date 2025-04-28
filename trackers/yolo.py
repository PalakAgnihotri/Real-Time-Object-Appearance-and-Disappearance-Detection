# detector/yolo.py

from ultralytics import YOLO

class YoloDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)  # CPU by default
        self.model.fuse()

    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]
        detections = []
        for r in results.boxes:
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            conf = float(r.conf[0])
            cls = int(r.cls[0])
            detections.append([x1, y1, x2, y2, conf, cls])
        return detections

