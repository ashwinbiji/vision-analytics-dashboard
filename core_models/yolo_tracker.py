import cv2
from ultralytics import YOLO

class ObjectTracker:
    def __init__(self):
        # Initializes the nano model. It will auto-download the 'yolov8n.pt' weight file on first run.
        self.model = YOLO('yolov8n.pt') 

    def process_frame(self, frame):
        # Run inference on the current frame
        results = self.model(frame, verbose=False)
        
        person_count = 0
        
        # Parse the results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Class 0 in the COCO dataset is 'person'
                if int(box.cls[0]) == 0:
                    person_count += 1
                    
                    # Extract coordinates and draw the bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Person", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame, person_count