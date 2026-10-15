import streamlit as st
import cv2
import tempfile
import pandas as pd
import time
from core_models.yolo_tracker import ObjectTracker
from core_models.gesture_tracker import GestureTracker
from core_models.heatmap_tracker import HeatmapTracker
from core_models.demographics_tracker import DemographicsTracker

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="VisionAI | Studio", page_icon="🧿", layout="wide", initial_sidebar_state="expanded")

# 2. ULTIMATE CYBER-GLASS CSS INJECTION
st.markdown("""
<style>
    /* Base UI Cleanup & Deep Dark Theme (Header restored for sidebar toggle) */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .block-container {
        padding: 1.5rem 3rem !important;
        max-width: 100% !important;
        background-color: #050505;
        background-image: 
            radial-gradient(at 0% 0%, rgba(17, 24, 39, 1) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(13, 0, 36, 1) 0px, transparent 50%);
        background-attachment: fixed;
        color: #ffffff;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', 'SF Pro Display', sans-serif;
    }

    /* Mesmerizing Animated Header */
    .hero-title {
        font-size: 3.5rem;
        font-weight: 900;
        letter-spacing: -1px;
        background: linear-gradient(to right, #00f2fe, #4facfe, #a18cd1, #00f2fe);
        background-size: 200% auto;
        color: #fff;
        background-clip: text;
        text-fill-color: transparent;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 4s linear infinite;
        margin-bottom: 0.2rem;
    }
    @keyframes shine {
        to { background-position: 200% center; }
    }
    
    .hero-subtitle {
        color: #8b9eb7;
        font-size: 1.1rem;
        font-weight: 400;
        letter-spacing: 0.5px;
        margin-bottom: 2rem;
    }

    /* Sidebar - Sleek Frosted Glass */
    [data-testid="stSidebar"] {
        background: rgba(10, 10, 15, 0.7) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #e2e8f0;
    }

    /* Bento Box Metric Cards */
    div[data-testid="metric-container"] {
        background: rgba(20, 20, 30, 0.4);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 1.5rem 2rem;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
    }
    
    /* Neon Glow on Hover */
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px) scale(1.02);
        border-color: rgba(0, 242, 254, 0.6);
        box-shadow: 0 15px 40px -5px rgba(0, 242, 254, 0.2), inset 0 0 20px rgba(0, 242, 254, 0.05);
    }
    
    [data-testid="stMetricValue"] {
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        color: #ffffff !important;
        text-shadow: 0 0 20px rgba(255,255,255,0.2);
    }
    [data-testid="stMetricLabel"] {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        color: #94a3b8 !important;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    /* Animated Live Indicator */
    .live-badge {
        display: inline-flex;
        align-items: center;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 6px 14px;
        border-radius: 20px;
        color: #10b981;
        font-weight: 600;
        font-size: 0.9rem;
        letter-spacing: 1px;
        margin-bottom: 1rem;
    }
    .pulse-dot {
        height: 8px;
        width: 8px;
        background-color: #10b981;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
        box-shadow: 0 0 10px #10b981;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(0.95); opacity: 1; }
        50% { transform: scale(1.2); opacity: 0.7; }
        100% { transform: scale(0.95); opacity: 1; }
    }

    /* Video Player Container Glow */
    .stImage > img {
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 20px 50px rgba(0,0,0,0.6);
        transition: border-color 0.3s ease;
    }
    .stImage > img:hover {
        border-color: rgba(161, 140, 209, 0.5);
    }

    /* Futuristic Buttons */
    .stDownloadButton > button, .stButton > button {
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        color: #ffffff !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    .stDownloadButton > button:hover, .stButton > button:hover {
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%) !important;
        border-color: transparent !important;
        box-shadow: 0 10px 20px rgba(0, 242, 254, 0.3) !important;
        transform: translateY(-2px) !important;
        color: #000 !important;
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
        .hero-title { font-size: 2.2rem; }
        .block-container { padding: 1rem !important; }
        [data-testid="stMetricValue"] { font-size: 2.5rem !important; }
    }
</style>
""", unsafe_allow_html=True)

# 3. HERO HEADER
st.markdown('<div class="hero-title">VisionAI Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Next-Generation Spatial Intelligence & Edge Demographics Interface</div>', unsafe_allow_html=True)

# 4. SIDEBAR (Command Center)
with st.sidebar:
    st.markdown("### 🎛️ Neural Engine")
    
    app_mode = st.selectbox(
        "Select Vision Pipeline", 
        [
            "Demographics (Age & Gender)",
            "Crowd Counting (YOLOv8)", 
            "Gesture Control (MediaPipe)", 
            "Spatial Heatmap (Density)"
        ]
    )
    
    tripwire_dir = "Horizontal"
    if app_mode == "Crowd Counting (YOLOv8)":
        tripwire_dir = st.radio("Tripwire Axis", ["Horizontal", "Vertical"], horizontal=True)
        
    st.markdown("### 📁 Data Ingestion")
    uploaded_video = st.file_uploader("Drop MP4/MOV here", type=['mp4', 'mov', 'avi'])
    
    st.markdown("---")
    st.caption("⚡ **Architecture:** Serverless Edge Inference\n\n🔒 **Privacy:** Zero-retention pipeline")

# 5. MAIN WORKSPACE
if uploaded_video is None:
    # Mesmerizing Empty State
    st.markdown("<br><br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("#### 🎯 1. Select Engine\nInitialize YOLOv8, MediaPipe, or Demographic routing in the command center.")
    with c2:
        st.warning("#### ⚙️ 2. Calibrate\nAdjust spatial tripwires and tracking logic to match your environment.")
    with c3:
        st.success("#### 🚀 3. Deploy\nUpload your telemetry video to begin real-time edge processing.")
else:
    # 6. ACTIVE DASHBOARD UI
    tab1, tab2 = st.tabs(["🔴 Live Telemetry", "📊 Data Export"])
    
    with tab1:
        dash_col1, dash_col2 = st.columns([7, 3], gap="large")
        
        with dash_col1:
            st.markdown('<div class="live-badge"><span class="pulse-dot"></span> LIVE NEURAL FEED</div>', unsafe_allow_html=True)
            stframe = st.empty()
            progress_bar = st.progress(0)
            
        with dash_col2:
            st.markdown("<br>", unsafe_allow_html=True)
            metric_placeholder = st.empty()
            
            st.markdown("<br>", unsafe_allow_html=True)
            status_box = st.empty()
            status_box.info(f"⚡ **System:** Initializing Weights...\n\n🧠 **Pipeline:** `{app_mode}`")

    # 7. BACKEND PROCESSING LOOP
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    cap = cv2.VideoCapture(tfile.name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if app_mode == "Crowd Counting (YOLOv8)":
        tracker = ObjectTracker(direction=tripwire_dir)
        metric_label = "Total Subjects"
    elif app_mode == "Gesture Control (MediaPipe)":
        tracker = GestureTracker()
        metric_label = "Active Gesture"
    elif app_mode == "Spatial Heatmap (Density)":
        tracker = HeatmapTracker()
        metric_label = "Area Density"
    else:
        tracker = DemographicsTracker()
        metric_label = "People In Frame"

    analytics_data = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        if frame_count % 3 != 0:
            continue
            
        if total_frames > 0:
            progress_bar.progress(min(frame_count / total_frames, 1.0))
            
        # Neural Network Execution
        processed_frame, metric = tracker.process_frame(frame)
        analytics_data.append({"Frame": frame_count, metric_label: metric})

        # Render Video
        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        stframe.image(rgb_frame, channels="RGB", width="stretch")
        
        # Render Metrics
        display_val = str(metric)
        if app_mode == "Gesture Control (MediaPipe)":
            display_val = "PINCH!" if metric == 1 else "None"
            
        metric_placeholder.metric(label=metric_label, value=display_val)
        status_box.success(f"⚡ **Status:** Real-time analysis ({(frame_count/total_frames)*100:.1f}%)\n\n🧠 **Pipeline:** `{app_mode}`")

    cap.release()
    progress_bar.progress(1.0)
    status_box.success(f"✅ **Status:** Inference Complete.\n\n🧠 **Pipeline:** `{app_mode}`")

    # 8. DATA EXPORT TAB
    with tab2:
        st.markdown("### Processed Telemetry Nodes")
        if len(analytics_data) > 0:
            df = pd.DataFrame(analytics_data)
            
            data_col1, data_col2 = st.columns([3, 1])
            with data_col1:
                st.dataframe(df, use_container_width=True, height=400)
                
            with data_col2:
                st.markdown("#### Database Hub")
                st.caption("Extract this payload for seamless integration with external dashboarding tools.")
                csv = df.to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    label="⬇️ Export CSV Payload",
                    data=csv,
                    file_name=f"visionAI_{app_mode.split()[0].lower()}_export.csv",
                    mime="text/csv",
                    use_container_width=True
                )