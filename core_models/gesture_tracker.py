import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

class GestureTracker:
    def __init__(self):
        # Use MediaPipe Tasks API for Hand Landmark Detection
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

        if self.use_tasks_api:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = self.detector.detect(mp_image)

            if detection_result.hand_landmarks:
                for hand_landmarks in detection_result.hand_landmarks:
                    # Extract landmark coordinates (Thumb tip: index 4, Index finger tip: index 8)
                    thumb_tip = hand_landmarks[4]
                    index_tip = hand_landmarks[8]
                    
                    tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)
                    ix, iy = int(index_tip.x * w), int(index_tip.y * h)
                    
                    # Draw points and connecting line
                    cv2.circle(frame, (tx, ty), 8, (0, 0, 255), -1)
                    cv2.circle(frame, (ix, iy), 8, (0, 0, 255), -1)
                    cv2.line(frame, (tx, ty), (ix, iy), (255, 0, 255), 3)

                    # Calculate Euclidean distance for pinch gesture detection
                    distance = np.hypot(tx - ix, ty - iy)
                    if distance < 40:
                        gesture_status = "Pinch Detected!"
                        cv2.putText(frame, "PINCH", (min(tx, ix), min(ty, iy) - 15), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(frame, f"Gesture Module | Status: {gesture_status}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        return frame, 1 if gesture_status == "Pinch Detected!" else 0