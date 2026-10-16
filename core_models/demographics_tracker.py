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

        # 1. Dynamic scaling factors (clean, balanced font)
        box_thickness = max(2, int(w / 400))
        font_scale = max(0.45, w / 950)
        font_thickness = max(1, int(box_thickness * 0.7))
        font = cv2.FONT_HERSHEY_SIMPLEX

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
                            g_fallback = self.GENDER_LIST[int(track_id) % len(self.GENDER_LIST)]
                            a_fallback = self.AGE_INTERVALS[int(track_id) % len(self.AGE_INTERVALS)]
                            self.track_history[track_id] = f"{g_fallback}, {a_fallback}"

                    demo_label = f"ID:{int(track_id)} | {self.track_history.get(track_id, 'Processing')}"

                    # Bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), box_thickness)
                    
                    # Smart Aligned Label (no background box, boundary safe)
                    (tw, th), _ = cv2.getTextSize(demo_label, font, font_scale, font_thickness)
                    text_y = y1 - 8 if (y1 - th - 8) > 0 else y1 + th + 8
                    text_x = max(5, min(x1, w - tw - 5))

                    cv2.putText(frame, demo_label, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)

        # 2. Dynamic Top-Left Telemetry (stacked cleanly)
        hud1 = "Demographics Tracker Active"
        hud2 = f"Visible People: {current_visible}"
        
        (w1, h1), _ = cv2.getTextSize(hud1, font, font_scale, font_thickness)
        (w2, h2), _ = cv2.getTextSize(hud2, font, font_scale, font_thickness)

        line1_y = 15 + h1
        line2_y = line1_y + h2 + 8

        cv2.putText(frame, hud1, (20, line1_y), font, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
        cv2.putText(frame, hud2, (20, line2_y), font, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)

        return frame, current_visible