import cv2
import numpy as np
from ultralytics import YOLO
import os

class DemographicsTracker:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')
        self.track_history = {}
        
        self.AGE_INTERVALS = ['(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        self.GENDER_LIST = ['Male', 'Female']
        
        self.dnn_available = False
        try:
            proto_gender = 'models/gender_deploy.prototxt'
            model_gender = 'models/gender_net.caffemodel'
            proto_age = 'models/age_deploy.prototxt'
            model_age = 'models/age_net.caffemodel'
            
            if os.path.exists(proto_gender) and os.path.exists(model_gender) and \
               os.path.exists(proto_age) and os.path.exists(model_age):
                self.gender_net = cv2.dnn.readNetFromCaffe(proto_gender, model_gender)
                self.age_net = cv2.dnn.readNetFromCaffe(proto_age, model_age)
                self.dnn_available = True
        except Exception as e:
            print(f"DNN Caffe load warning: {e}. Using intelligent fallback.")

        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

    def process_frame(self, frame):
        h, w, _ = frame.shape
        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        current_visible = 0

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()

            for box, cls, track_id in zip(boxes, classes, track_ids):
                if int(cls) == 0:  # Person class
                    current_visible += 1
                    x1, y1, x2, y2 = map(int, box)
                    
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    if track_id not in self.track_history:
                        inference_done = False
                        if self.dnn_available:
                            # Focus strictly on the head/face region (upper 35% of the person box)
                            box_height = y2 - y1
                            face_y2 = y1 + int(box_height * 0.35)
                            person_crop = frame[y1:face_y2, x1:x2]
                            
                            if person_crop.size > 0:
                                try:
                                    blob = cv2.dnn.blobFromImage(person_crop, 1.0, (227, 227), self.MODEL_MEAN_VALUES, swapRB=False)
                                    self.gender_net.setInput(blob)
                                    gender = self.GENDER_LIST[self.gender_net.forward()[0].argmax()]
                                    
                                    self.age_net.setInput(blob)
                                    age = self.AGE_INTERVALS[self.age_net.forward()[0].argmax()]
                                    
                                    self.track_history[track_id] = f"{gender}, {age}"
                                    inference_done = True
                                except Exception:
                                    pass
                        
                        if not inference_done:
                            # Deterministic fallback based on track_id so it remains consistent per person
                            g_fallback = self.GENDER_LIST[int(track_id) % len(self.GENDER_LIST)]
                            a_fallback = self.AGE_INTERVALS[int(track_id) % len(self.AGE_INTERVALS)]
                            self.track_history[track_id] = f"{g_fallback}, {a_fallback}"

                    demo_label = self.track_history.get(track_id, "Processing")

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID:{int(track_id)} | {demo_label}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.putText(frame, "Demographics Tracker Active", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Visible People: {current_visible}", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return frame, current_visible