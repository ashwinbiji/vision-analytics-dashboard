import streamlit as st
import cv2
import tempfile
import pandas as pd
from core_models.yolo_tracker import ObjectTracker
from core_models.gesture_tracker import GestureTracker

st.set_page_config(page_title="Vision Analytics Dashboard", layout="wide")

st.title("Real-Time Computer Vision Analytics")
st.sidebar.header("Control Panel")

app_mode = st.sidebar.selectbox(
    "Choose Analytics Module", 
    ["Crowd Counting (YOLOv8)", "Gesture Control (MediaPipe)"]
)

uploaded_video = st.sidebar.file_uploader("Upload Video (.mp4, .mov)", type=['mp4', 'mov', 'avi'])

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    
    cap = cv2.VideoCapture(tfile.name)
    
    if app_mode == "Crowd Counting (YOLOv8)":
        tracker = ObjectTracker()
    else:
        tracker = GestureTracker()

    stframe = st.empty()
    analytics_placeholder = st.empty()
    
    analytics_data = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        if frame_count % 3 != 0:
            continue
            
        processed_frame, metric = tracker.process_frame(frame)
        
        metric_label = "People Count" if app_mode == "Crowd Counting (YOLOv8)" else "Gesture Status"
        analytics_data.append({"Frame": frame_count, metric_label: metric})

        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        
        stframe.image(rgb_frame, channels="RGB", use_container_width=True)
        analytics_placeholder.metric(label=metric_label, value=metric)

    cap.release()
    st.success("Video processing complete!")

    if len(analytics_data) > 0:
        df = pd.DataFrame(analytics_data)
        st.subheader("Analytics Data")
        st.dataframe(df.head(10))
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Analytics CSV",
            data=csv,
            file_name="vision_analytics.csv",
            mime="text/csv",
        )
else:
    st.info("Please upload a video clip in the sidebar to begin processing.")
