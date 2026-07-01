from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_name='yolov8n.pt'):
        # Load YOLO model
        self.model = YOLO(model_name)
        # Class 0: person, Class 67: cell phone
        self.target_classes = [0, 67] 

    def detect(self, frame):
        results = self.model(frame, verbose=False)
        person_count = 0
        phone_detected = False
        bboxes = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id in self.target_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    
                    if cls_id == 0:
                        person_count += 1
                        bboxes.append((x1, y1, x2, y2, "Person", conf))
                    elif cls_id == 67:
                        phone_detected = True
                        bboxes.append((x1, y1, x2, y2, "Phone", conf))

        return person_count, phone_detected, bboxes
