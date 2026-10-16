import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

class GestureTracker:
    def __init__(self):
        base_options = python.BaseOptions(model_asset_path='models/hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5
        )
        try:
            self.detector = vision.HandLandmarker.create_from_options(options)
            self.use_tasks_api = True
        except Exception:
            self.use_tasks_api = False

    def process_frame(self, frame):
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gesture_status = "No Gesture"

        # Dynamic scaling factors
        box_thickness = max(2, int(w / 400))
        font_scale = max(0.45, w / 950)
        font_thickness = max(1, int(box_thickness * 0.7))
        circle_radius = max(5, int(w / 200))
        font = cv2.FONT_HERSHEY_SIMPLEX

        if self.use_tasks_api:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = self.detector.detect(mp_image)

            if detection_result.hand_landmarks:
                for hand_landmarks in detection_result.hand_landmarks:
                    thumb_tip = hand_landmarks[4]
                    index_tip = hand_landmarks[8]
                    
                    tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)
                    ix, iy = int(index_tip.x * w), int(index_tip.y * h)
                    
                    cv2.circle(frame, (tx, ty), circle_radius, (0, 0, 255), -1)
                    cv2.circle(frame, (ix, iy), circle_radius, (0, 0, 255), -1)
                    cv2.line(frame, (tx, ty), (ix, iy), (255, 0, 255), box_thickness)

                    distance = np.hypot(tx - ix, ty - iy)
                    if distance < (w * 0.035):
                        gesture_status = "Pinch Detected!"
                        
                        # Smart Aligned Pinch Label
                        pinch_text = "PINCH"
                        (pw, ph), _ = cv2.getTextSize(pinch_text, font, font_scale, font_thickness)
                        px, py = min(tx, ix), min(ty, iy)
                        text_y = py - 8 if (py - ph - 8) > 0 else py + ph + 8
                        text_x = max(5, min(px, w - pw - 5))

                        cv2.putText(frame, pinch_text, (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)

        # Dynamic Top-Left Telemetry
        hud1 = f"Gesture Module | Status: {gesture_status}"
        (w1, h1), _ = cv2.getTextSize(hud1, font, font_scale, font_thickness)
        line1_y = 15 + h1

        cv2.putText(frame, hud1, (20, line1_y), font, font_scale, (255, 255, 0), font_thickness, cv2.LINE_AA)

        return frame, 1 if gesture_status == "Pinch Detected!" else 0