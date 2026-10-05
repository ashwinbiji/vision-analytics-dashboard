
# 👁️ Real-Time Vision Analytics Dashboard

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-00FFFF?logo=yolo&logoColor=black)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An interactive, edge-computing computer vision platform built to process video streams in real-time. This application performs automated crowd tracking and hand gesture detection, exporting raw inference metrics into structured CSV analytics. 

Engineered for resource-constrained environments (developed natively on a Linux), the architecture optimizes CPU inference loops and handles asynchronous UI updates statelessly.

🚀 **Live Cloud Demo:** [View on Hugging Face Spaces](https://huggingface.co/spaces/YOUR_HF_USERNAME/vision-analytics-dashboard)

---

## 🛠️ Tech Stack & System Architecture

| Domain | Technology | Implementation Details |
|---|---|---|
| **Object Detection** | Ultralytics YOLOv8 (Nano) | Real-time object identification optimized for CPU inference. |
| **Tracking Algorithm** | ByteTrack | Assigns unique IDs across continuous frames to prevent duplicate counting. |
| **Gesture Tracking** | Google MediaPipe | 21-point 3D hand landmark tracking utilizing Euclidean distance calculations. |
| **Frontend UI** | Streamlit | Responsive, stateless dashboard layout managing temporary file ingestion. |
| **Data Processing** | OpenCV & Pandas | Frame extraction, vector geometry, and CSV telemetry generation. |

---

## 🚀 Core Features

* **Multi-Directional Virtual Tripwires:** Engineered Cartesian vector logic to track when a bounding box centroid crosses a user-defined threshold. Supports both Horizontal (forward/backward movement) and Vertical (lateral movement) orientations.
* **Stateful Object Tracking:** Integrates ByteTrack to maintain object identity across frames, ensuring enterprise-grade accuracy in crowd counting.
* **Hand Gesture Recognition:** Tracks skeletal hand joints and calculates the geometric distance between thumb and index finger tips to register input gestures (e.g., pinch detection).
* **Telemetry Data Export:** Converts real-time frame telemetry into structured Pandas DataFrames with one-click CSV export functionality for downstream data pipelines.
* **Frame Optimization:** Processes every $N$-th frame to maximize throughput and ensure consistent FPS performance on CPU-bound hardware.

---
## 📂 Project Architecture

```text
vision-analytics-dashboard/
├── app.py                     # Main Streamlit UI frontend and loop controller
├── requirements.txt           # Project dependencies
├── core_models/
│   ├── yolo_tracker.py        # YOLOv8 + ByteTrack & tripwire logic
│   ├── gesture_tracker.py     # MediaPipe hand landmark detection
│   └── heatmap_tracker.py     # NumPy spatial density matrix accumulation
└── README.md                  # Project documentation
```
## 💻 Local Installation & Setup

**1. Clone the repository:**
`bash
git clone https://github.com/YOUR_USERNAME/vision-analytics-dashboard.git
cd vision-analytics-dashboard
`

**2. Initialize the virtual environment:**
`bash
python3 -m venv cv_env
source cv_env/bin/activate
`

**3. Install dependencies:**
`bash
pip install -r requirements.txt
`

**4. Launch the dashboard:**
`bash
python3 -m streamlit run app.py
`

---

## 📊 Usage Guide

1. Launch the Streamlit application.
2. Select an analytics module from the sidebar (**Crowd Counting** or **Gesture Control**).
3. If using Crowd Counting, select the **Tripwire Direction** that matches the movement axis of your video.
4. Upload a `.mp4` or `.mov` file. The inference engine will begin processing immediately.
5. Download the final `vision_analytics.csv` to review the frame-by-frame telemetry.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
EOF
