from ultralytics import YOLO
import cv2
import numpy as np
from .config import settings


# load model once
model = YOLO(settings.YOLO_MODEL_PATH)


# map some YOLO class ids/names to pantry-friendly labels if needed


def run_yolo_on_image_bytes(image_bytes: bytes, conf_thresh=0.3):
    # image_bytes: raw bytes from upload
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    results = model.predict(img, conf=conf_thresh, imgsz=640)
    # results is a list; take first
    out = []
    r = results[0]
    for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
        x1, y1, x2, y2 = [int(x) for x in box.tolist()]
        label = r.names[int(cls)]
        out.append({
        "name": label,
        "bbox": [x1, y1, x2, y2],
        "conf": float(conf)
        })
    return out