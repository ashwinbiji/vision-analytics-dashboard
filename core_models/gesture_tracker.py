import cv2
import mediapipe as mp
import math

class GestureTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils

    def process_frame(self, frame):
        # Convert BGR (OpenCV format) to RGB (MediaPipe format)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        gesture_status = "Open Hand"

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw the skeletal connections
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Extract coordinates for Thumb Tip (4) and Index Finger Tip (8)
                h, w, _ = frame.shape
                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]
                
                cx_thumb, cy_thumb = int(thumb_tip.x * w), int(thumb_tip.y * h)
                cx_index, cy_index = int(index_tip.x * w), int(index_tip.y * h)
                
                # Calculate Euclidean distance between the two points
                distance = math.hypot(cx_index - cx_thumb, cy_index - cy_thumb)
                
                # If the fingers are close together, trigger the pinch
                if distance < 40:
                    gesture_status = "Pinch Detected!"
                    cv2.circle(frame, (cx_index, cy_index), 15, (0, 0, 255), cv2.FILLED)

        return frame, gesture_status