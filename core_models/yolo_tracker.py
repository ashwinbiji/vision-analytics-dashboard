import cv2
from ultralytics import YOLO

class ObjectTracker:
    def __init__(self, direction="Horizontal"):
        self.model = YOLO('yolov8n.pt')
        self.track_history = {}
        self.crossed_count = 0
        self.counted_ids = set()
        self.direction = direction

    def process_frame(self, frame):
        h, w, _ = frame.shape
        
        # Set line position based on direction
        if self.direction == "Horizontal":
            line_pos = int(h * 0.6)  # 60% down the screen
            cv2.line(frame, (0, line_pos), (w, line_pos), (255, 0, 0), 2)
            cv2.putText(frame, "TRIPWIRE (Horizontal)", (10, line_pos - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        else:
            line_pos = int(w * 0.5)  # 50% across the screen
            cv2.line(frame, (line_pos, 0), (line_pos, h), (255, 0, 0), 2)
            cv2.putText(frame, "TRIPWIRE (Vertical)", (line_pos + 10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        current_visible = 0

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()

            for box, cls, track_id in zip(boxes, classes, track_ids):
                if int(cls) == 0:  # Class 0 = Person
                    current_visible += 1
                    x1, y1, x2, y2 = map(int, box)
                    
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(frame, f"ID: {int(track_id)}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Line Crossing Logic
                    if track_id in self.track_history:
                        prev_cx, prev_cy = self.track_history[track_id]
                        
                        crossed = False
                        if self.direction == "Horizontal":
                            if (prev_cy < line_pos <= cy) or (prev_cy > line_pos >= cy):
                                crossed = True
                        else:
                            if (prev_cx < line_pos <= cx) or (prev_cx > line_pos >= cx):
                                crossed = True

                        if crossed and track_id not in self.counted_ids:
                            self.crossed_count += 1
                            self.counted_ids.add(track_id)
                            
                            # Highlight line on crossing
                            if self.direction == "Horizontal":
                                cv2.line(frame, (0, line_pos), (w, line_pos), (0, 255, 255), 4)
                            else:
                                cv2.line(frame, (line_pos, 0), (line_pos, h), (0, 255, 255), 4)

                    self.track_history[track_id] = (cx, cy)

        cv2.putText(frame, f"Visible Now: {current_visible}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Total Crossed: {self.crossed_count}", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        return frame, self.crossed_count