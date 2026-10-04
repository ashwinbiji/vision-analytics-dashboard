import streamlit as st
import cv2
import tempfile
import pandas as pd
from core_models.yolo_tracker import ObjectTracker
from core_models.gesture_tracker import GestureTracker

# Configure the Streamlit page layout
st.set_page_config(page_title="Vision Analytics Dashboard", layout="wide")

st.title("Real-Time Computer Vision Analytics")
st.sidebar.header("Control Panel")

# 1. Sidebar Navigation
app_mode = st.sidebar.selectbox(
    "Choose Analytics Module", 
    ["Crowd Counting (YOLOv8)", "Gesture Control (MediaPipe)"]
)

# 2. File Ingestion
uploaded_video = st.sidebar.file_uploader("Upload Video (.mp4, .mov)", type=['mp4', 'mov', 'avi'])

if uploaded_video is not None:
    # Save the uploaded video to a temporary file so OpenCV can read it
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    
    cap = cv2.VideoCapture(tfile.name)
    
    # Initialize the correct tracker
    if app_mode == "Crowd Counting (YOLOv8)":
        tracker = ObjectTracker()
    else:
        tracker = GestureTracker()

    # Create empty placeholders in the UI to update dynamically
    stframe = st.empty()
    analytics_placeholder = st.empty()
    
    analytics_data = []
    frame_count = 0

    # 3. Video Processing Loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Performance optimization: Process every 3rd frame
        if frame_count % 3 != 0:
            continue
            
        # Run inference
        if app_mode == "Crowd Counting (YOLOv8)":
            processed_frame, metric = tracker.process_frame(frame)
            metric_label = "People Count"
            analytics_data.append({"Frame": frame_count, "People_Detected": metric})
        else:
            processed_frame, metric = tracker.process_frame(frame)
            metric_label = "Gesture Status"
            analytics_data.append({"Frame": frame_count, "Gesture": metric})

        # Convert OpenCV BGR format to Streamlit's native RGB format
        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        
        # Render the frame and update the live metric
        stframe.image(rgb_frame, channels="RGB", use_container_width=True)
        analytics_placeholder.metric(label=metric_label, value=metric)

    cap.release()
    st.success("Video processing complete!")

    # 4. Analytics Export
    if len(analytics_data) > 0:
        df = pd.DataFrame(analytics_data)
        st.subheader("Analytics Data")
        st.dataframe(df.head(10)) # Show preview
        
        # Convert DataFrame to CSV for download
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Analytics CSV",
            data=csv,
            file_name="vision_analytics.csv",
            mime="text/csv",
        )
else:
    st.info("Please upload a video clip in the sidebar to begin processing.")