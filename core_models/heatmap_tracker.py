import cv2
import numpy as np
from ultralytics import YOLO

class HeatmapTracker:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')
        self.heatmap_matrix = None
        self.max_traffic = 1

    def process_frame(self, frame):
        h, w, _ = frame.shape
        
        # 1. Dynamic scaling factors
        box_thickness = max(2, int(w / 400))
        font_scale = max(0.45, w / 950)
        font_thickness = max(1, int(box_thickness * 0.7))
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Initialize accumulation matrix on first frame
        if self.heatmap_matrix is None:
            self.heatmap_matrix = np.zeros((h, w), dtype=np.float32)

        results = self.model(frame, verbose=False)
        current_visible = 0

        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()

            for box, cls in zip(boxes, classes):
                if int(cls) == 0:  # Class 0 = Person
                    current_visible += 1
                    x1, y1, x2, y2 = map(int, box)
                    
                    cx = int((x1 + x2) / 2)
                    cy = int(y2)  # Focus on feet placement for heat intensity

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), box_thickness)

                    # Increment weight on accumulation matrix
                    if 0 <= cx < w and 0 <= cy < h:
                        self.heatmap_matrix[cy, cx] += 5.0

        # Apply Gaussian blur to smooth into heat zones
        blurred_heatmap = cv2.GaussianBlur(self.heatmap_matrix, (55, 55), 0)
        
        if blurred_heatmap.max() > 0:
            normalized_heatmap = cv2.normalize(blurred_heatmap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        else:
            normalized_heatmap = blurred_heatmap

        heatmap_8bit = np.uint8(normalized_heatmap)
        color_heatmap = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_JET)

        # Blend heatmap with video frame
        blended_frame = cv2.addWeighted(frame, 0.6, color_heatmap, 0.4, 0)

        # 2. Dynamic Top-Left Telemetry (relative vertical stacking)
        hud1 = "Spatial Heatmap Active"
        hud2 = f"Visible People: {current_visible}"
        
        (w1, h1), _ = cv2.getTextSize(hud1, font, font_scale, font_thickness)
        (w2, h2), _ = cv2.getTextSize(hud2, font, font_scale, font_thickness)

        line1_y = 15 + h1
        line2_y = line1_y + h2 + 8

        cv2.putText(blended_frame, hud1, (20, line1_y), font, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)
        cv2.putText(blended_frame, hud2, (20, line2_y), font, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)

        return blended_frame, current_visible