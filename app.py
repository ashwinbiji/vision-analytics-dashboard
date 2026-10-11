import streamlit as st
import cv2
import tempfile
import pandas as pd
from core_models.yolo_tracker import ObjectTracker
from core_models.gesture_tracker import GestureTracker
from core_models.heatmap_tracker import HeatmapTracker
from core_models.demographics_tracker import DemographicsTracker

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="VisionAI | Dashboard", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

# 2. ADVANCED RESPONSIVE CSS INJECTION
st.markdown("""
<style>
    /* Base UI Cleanup */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Default Desktop Padding */
    .block-container {
        padding: 2rem 4rem !important;
        max-width: 100% !important;
        transition: padding 0.3s ease;
    }
    
    /* Global Typography */
    html, body, [class*="css"]  {
        font-family: 'Inter', system-ui, sans-serif;
    }
    
    /* Responsive Gradient Header */
    .saas-header {
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        line-height: 1.2;
    }
    .saas-subtitle {
        color: #8b9eb7;
        font-size: 1.1rem;
        font-weight: 500;
        margin-bottom: 2rem;
    }
    
    /* Glassmorphic Metric Cards */
    div[data-testid="metric-container"] {
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-4px);
        border-color: rgba(0, 242, 254, 0.5);
        box-shadow: 0 10px 30px -5px rgba(0, 242, 254, 0.2);
    }
    [data-testid="stMetricValue"] {
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        color: #ffffff !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: #94a3b8 !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    
    /* Live Pulse Animation */
    .pulse-dot {
        height: 12px;
        width: 12px;
        background-color: #10b981;
        border-radius: 50%;
        display: inline-block;
        animation: pulse-animation 2s infinite;
        margin-right: 8px;
    }
    @keyframes pulse-animation {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    /* Buttons */
    .stDownloadButton > button {
        width: 100%;
        background: linear-gradient(135deg, #0284c7 0%, #00f2fe 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.8rem 1rem !important;
        transition: all 0.2s ease;
    }
    .stDownloadButton > button:hover {
        opacity: 0.9;
        transform: scale(1.02);
    }

    /* ==== MEDIA QUERIES FOR TRUE RESPONSIVENESS ==== */
    
    /* Tablets (Max Width: 1024px) */
    @media (max-width: 1024px) {
        .block-container {
            padding: 2rem !important;
        }
        .saas-header {
            font-size: 2.5rem;
        }
        [data-testid="stMetricValue"] {
            font-size: 2.8rem !important;
        }
    }
    
    /* Mobile Devices (Max Width: 768px) */
    @media (max-width: 768px) {
        .block-container {
            padding: 1rem !important;
        }
        .saas-header {
            font-size: 2rem;
            text-align: center;
        }
        .saas-subtitle {
            text-align: center;
            font-size: 0.95rem;
        }
        div[data-testid="metric-container"] {
            padding: 1rem;
            text-align: center;
        }
        [data-testid="stMetricValue"] {
            font-size: 2.2rem !important;
        }
        /* Make tabs stack properly on small screens */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            flex-wrap: wrap;
        }
    }
</style>
""", unsafe_allow_html=True)

# 3. HEADER
st.markdown('<div class="saas-header">VisionAI Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="saas-subtitle">Next-Generation Spatial Intelligence & Demographic Analytics | <b>v4.1.0</b></div>', unsafe_allow_html=True)

# 4. SIDEBAR (Control Center)
with st.sidebar:
    st.markdown("### 🎛️ Pipeline Setup")
    
    app_mode = st.selectbox(
        "Select Vision Engine", 
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
    st.caption("🔒 **Edge Processing:** Zero data retention.")

# 5. MAIN WORKSPACE
if uploaded_video is None:
    # Responsive Empty State Dashboard
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("#### 1. Select Engine\nPick from YOLOv8, Demographic AI, or Gesture Mapping in the sidebar.")
    with c2:
        st.warning("#### 2. Configure Logic\nAdjust spatial boundaries or tracking axes for your specific use case.")
    with c3:
        st.success("#### 3. Upload Media\nDrop your video file into the ingestion zone to start processing.")
else:
    # 6. ACTIVE DASHBOARD UI (Responsive Tabs & Columns)
    tab1, tab2 = st.tabs(["🔴 Live Telemetry", "📊 Data Export"])
    
    with tab1:
        # Streamlit automatically stacks these columns vertically on mobile screens
        dash_col1, dash_col2 = st.columns([7, 3], gap="large")
        
        with dash_col1:
            st.markdown('<div style="display: flex; align-items: center; margin-bottom: 10px;"><span class="pulse-dot"></span><b>LIVE FEED</b></div>', unsafe_allow_html=True)
            stframe = st.empty()
            progress_bar = st.progress(0)
            
        with dash_col2:
            st.markdown("<b>INSTANT METRICS</b>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            metric_placeholder = st.empty()
            
            st.markdown("<br>", unsafe_allow_html=True)
            status_box = st.empty()
            status_box.info(f"⚡ **Status:** Initializing Model...\n\n🧠 **Engine:** `{app_mode}`")

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
            
        processed_frame, metric = tracker.process_frame(frame)
        analytics_data.append({"Frame": frame_count, metric_label: metric})

        # Render Video feed (width="stretch" prevents the warning)
        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        stframe.image(rgb_frame, channels="RGB", width="stretch")
        
        # Render Metrics
        display_val = str(metric)
        if app_mode == "Gesture Control (MediaPipe)":
            display_val = "PINCH!" if metric == 1 else "None"
            
        metric_placeholder.metric(label=metric_label, value=display_val)
        status_box.success(f"⚡ **Status:** Processing ({(frame_count/total_frames)*100:.1f}%)\n\n🧠 **Engine:** `{app_mode}`")

    cap.release()
    progress_bar.progress(1.0)
    status_box.success(f"✅ **Status:** Stream Complete.\n\n🧠 **Engine:** `{app_mode}`")

    # 8. DATA EXPORT TAB
    with tab2:
        st.markdown("### Processed Telemetry Records")
        if len(analytics_data) > 0:
            df = pd.DataFrame(analytics_data)
            
            data_col1, data_col2 = st.columns([3, 1])
            with data_col1:
                st.dataframe(df, use_container_width=True, height=400)
                
            with data_col2:
                st.markdown("#### Integration Hub")
                st.caption("Export the payload to integrate with Tableau, PowerBI, or an external database.")
                csv = df.to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    label="⬇️ Download CSV Payload",
                    data=csv,
                    file_name=f"visionAI_{app_mode.split()[0].lower()}_export.csv",
                    mime="text/csv",
                    use_container_width=True
                )