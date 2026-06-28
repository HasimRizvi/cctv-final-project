from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

def detect_incidents(frame):
    results = model(frame, verbose=False)
    detections = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if conf > 0.4:
                detections.append({
                    "label": label,
                    "confidence": conf,
                    "box": (x1, y1, x2, y2)
                })

    return detections, results[0].plot()