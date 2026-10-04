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
        
        # Initialize the accumulation matrix on the first frame
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
                    
                    # Calculate the center point (centroid) of the bounding box base
                    cx = int((x1 + x2) / 2)
                    cy = int(y2)  # Focus on foot placement for floor traffic

                    # Draw standard bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Increment weight on the accumulation matrix
                    if 0 <= cx < w and 0 <= cy < h:
                        self.heatmap_matrix[cy, cx] += 5.0  # Intensity boost per frame

        # Apply Gaussian blur to spread out the points into smooth heat zones
        blurred_heatmap = cv2.GaussianBlur(self.heatmap_matrix, (55, 55), 0)
        
        # Normalize and scale to 0-255 for color mapping
        if blurred_heatmap.max() > 0:
            normalized_heatmap = cv2.normalize(blurred_heatmap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        else:
            normalized_heatmap = blurred_heatmap

        # Convert to 8-bit image
        heatmap_8bit = np.uint8(normalized_heatmap)

        # Apply OpenCV thermal color map (JET: Blue -> Cyan -> Yellow -> Red)
        color_heatmap = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_JET)

        # Blend the heatmap over the original video frame (Transparency ratio: 60% video, 40% heatmap)
        blended_frame = cv2.addWeighted(frame, 0.6, color_heatmap, 0.4, 0)

        # Display metadata overlay
        cv2.putText(blended_frame, f"Spatial Heatmap Active", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(blended_frame, f"Visible People: {current_visible}", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return blended_frame, current_visible