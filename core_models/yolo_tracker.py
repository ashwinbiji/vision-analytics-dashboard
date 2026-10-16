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
        
        # 1. Dynamic scaling factors (cleaner, smaller font)
        box_thickness = max(2, int(w / 400))
        font_scale = max(0.45, w / 950)
        font_thickness = max(1, int(box_thickness * 0.7))
        font = cv2.FONT_HERSHEY_SIMPLEX

        # 2. Draw Tripwire Line & Label
        if self.direction == "Horizontal":
            line_pos = int(h * 0.6)
            cv2.line(frame, (0, line_pos), (w, line_pos), (255, 0, 0), box_thickness)
            
            trip_text = "TRIPWIRE (Horizontal)"
            (tw, th), _ = cv2.getTextSize(trip_text, font, font_scale * 0.8, font_thickness)
            trip_y = line_pos - 8 if (line_pos - th - 8) > 0 else line_pos + th + 12
            cv2.putText(frame, trip_text, (10, trip_y), font, font_scale * 0.8, (255, 0, 0), font_thickness, cv2.LINE_AA)
        else:
            line_pos = int(w * 0.5)
            cv2.line(frame, (line_pos, 0), (line_pos, h), (255, 0, 0), box_thickness)
            
            trip_text = "TRIPWIRE (Vertical)"
            (tw, th), _ = cv2.getTextSize(trip_text, font, font_scale * 0.8, font_thickness)
            cv2.putText(frame, trip_text, (line_pos + 10, 20 + th), font, font_scale * 0.8, (255, 0, 0), font_thickness, cv2.LINE_AA)

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

                    # Bounding box & center tracking point
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), box_thickness)
                    cv2.circle(frame, (cx, cy), max(4, int(box_thickness * 1.5)), (0, 0, 255), -1)

                    # Smart Aligned Label (prevents top/side boundary clipping)
                    id_label = f"ID: {int(track_id)}"
                    (lw, lh), _ = cv2.getTextSize(id_label, font, font_scale, font_thickness)
                    text_y = y1 - 8 if (y1 - lh - 8) > 0 else y1 + lh + 8
                    text_x = max(5, min(x1, w - lw - 5))

                    cv2.putText(frame, id_label, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)

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
                            
                            # Highlight tripwire on trigger
                            if self.direction == "Horizontal":
                                cv2.line(frame, (0, line_pos), (w, line_pos), (0, 255, 255), box_thickness + 2)
                            else:
                                cv2.line(frame, (line_pos, 0), (line_pos, h), (0, 255, 255), box_thickness + 2)

                    self.track_history[track_id] = (cx, cy)

        # 3. Dynamic Top-Left Telemetry (relative vertical stacking)
        hud1 = f"Visible Now: {current_visible}"
        hud2 = f"Total Crossed: {self.crossed_count}"
        
        (w1, h1), _ = cv2.getTextSize(hud1, font, font_scale, font_thickness)
        (w2, h2), _ = cv2.getTextSize(hud2, font, font_scale, font_thickness)

        line1_y = 15 + h1
        line2_y = line1_y + h2 + 8

        cv2.putText(frame, hud1, (20, line1_y), font, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)
        cv2.putText(frame, hud2, (20, line2_y), font, font_scale, (0, 255, 255), font_thickness, cv2.LINE_AA)

        return frame, self.crossed_count