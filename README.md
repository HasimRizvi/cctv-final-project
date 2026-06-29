# 🎥 CCTV-Based Crime & Accident Detection System

> **Real-Time Incident Analytics powered by YOLOv8 AI**  
> Automatically detects crimes, crowd incidents, and vehicle accidents from CCTV footage — and raises instant alerts.

[![Live Demo](https://img.shields.io/badge/🌐_Live_Demo-Streamlit_Cloud-FF4B4B?style=for-the-badge)](https://cctv-final-project-kj6cdx6imicabgjjgc6vys.streamlit.app)
[![GitHub](https://img.shields.io/badge/GitHub-HasimRizvi-181717?style=for-the-badge&logo=github)](https://github.com/HasimRizvi/cctv-final-project)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/AI-YOLOv8-00FFFF?style=for-the-badge)](https://ultralytics.com)

---

## 📌 Introduction

Every year, road accidents, crimes, and public incidents go undetected for critical minutes — simply because no one is watching the right camera at the right time.

This project proposes an **AI-powered real-time surveillance system** that:

- 📹 Accepts **CCTV video footage** or **live webcam** input
- 🧠 Runs **YOLOv8** object detection on every frame
- 🔍 **Classifies events** as crime, crowd incident, or vehicle accident
- 🚨 **Raises instant alerts** with a timestamped log
- 💻 Displays everything in a **beautiful web UI** accessible from any browser

> Our goal is to minimize the time between an incident occurring and help being dispatched — giving victims the best possible chance of survival.

---

## 🖼️ System Overview

```
📹 CCTV / Video Input
        ↓
⚙️  OpenCV Frame Processing
        ↓
🧠  YOLOv8 Object Detection
        ↓
🔍  Event Classification Engine
        ↓
🚨  Real-Time Alert + Log
        ↓
💻  Streamlit Web UI
```

---

## 🚀 Live Demo

👉 **[Click here to open the live app](https://cctv-final-project-kj6cdx6imicabgjjgc6vys.streamlit.app)**

### How to use:
1. Open the live URL in any browser
2. Select **"Upload Video"** from the sidebar
3. Upload any `.mp4` / `.avi` / `.mov` video file
4. Click **▶️ Start Detection**
5. Watch the AI detect objects and log alerts in real time!

---

## ✨ Key Features

| Feature | Description |
|---|---|
| ⚡ Real-Time Detection | Processes video frame-by-frame using YOLOv8 nano model |
| 🔫 Weapon Detection | Detects knives and dangerous objects — raises crime alert |
| 👥 Crowd Analysis | Flags abnormal gatherings when 4+ people detected |
| 🚗 Accident Detection | Monitors vehicles — flags cars, trucks, buses in danger zones |
| 📋 Alert Log | Timestamped log of all detected incidents |
| ☁️ Cloud Deployed | Live on Streamlit Cloud — no installation needed |
| 🌐 Webcam Support | Connect live CCTV or webcam directly |

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 (64-bit) |
| AI Model | YOLOv8 (Ultralytics) |
| Video Processing | OpenCV |
| Web UI | Streamlit |
| ML Framework | PyTorch |
| Deployment | Streamlit Cloud |
| Version Control | GitHub |

---

## 📂 Project Structure

```
cctv-final-project/
│
├── app.py           ← Main Streamlit UI
├── detector.py      ← YOLOv8 detection engine
├── alert.py         ← Event classification & alert logic
├── requirements.txt ← Python dependencies
└── yolov8n.pt       ← YOLOv8 nano model weights
```

---

## 💻 Run Locally — Step by Step

### Prerequisites
- Windows 10/11 (64-bit)
- Python 3.11 64-bit → [Download here](https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe)
- Git → [Download here](https://git-scm.com/download/win)

---

### Step 1 — Clone the Repository

```bash
git clone https://github.com/HasimRizvi/cctv-final-project.git
cd cctv-final-project
```

---

### Step 2 — Install Dependencies

```bash
pip install ultralytics opencv-python streamlit numpy pillow torch torchvision
```

> ⏳ This may take 5–10 minutes (PyTorch is large). Please wait!

---

### Step 3 — Run the App

```bash
streamlit run app.py
```

Your browser will automatically open at:
```
http://localhost:8501
```

---

## 📄 Full Source Code

### `detector.py` — YOLOv8 Detection Engine

```python
from ultralytics import YOLO
import cv2

# Load pretrained YOLOv8 nano model (auto-downloads on first run)
model = YOLO("yolov8n.pt")

def detect_incidents(frame):
    """
    Run YOLOv8 detection on a single video frame.
    Returns list of detections and annotated frame.
    """
    results = model(frame, verbose=False)
    detections = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if conf > 0.4:  # Only keep detections above 40% confidence
                detections.append({
                    "label": label,
                    "confidence": conf,
                    "box": (x1, y1, x2, y2)
                })

    return detections, results[0].plot()  # plot() draws bounding boxes
```

---

### `alert.py` — Event Classification & Alert Logic

```python
import time

last_alert_time = {}

def check_alert(label, cooldown=10):
    """
    Prevent repeated alerts for same event within cooldown seconds.
    Returns True if alert should be raised.
    """
    now = time.time()
    if label not in last_alert_time or (now - last_alert_time[label]) > cooldown:
        last_alert_time[label] = now
        return True
    return False

def play_alert_sound():
    # Audio disabled for cloud compatibility
    pass

def classify_event(detections):
    """
    Rule-based event classification.
    Returns event string or None if no incident detected.
    """
    labels = [d["label"] for d in detections]

    if "knife" in labels:
        return "⚠️ CRIME DETECTED: Weapon Spotted"
    elif len([l for l in labels if l == "person"]) >= 4:
        return "⚠️ CROWD INCIDENT: Multiple Persons"
    elif "car" in labels or "truck" in labels or "bus" in labels:
        return "🚗 VEHICLE DETECTED: Possible Accident Zone"
    else:
        return None
```

---

### `app.py` — Streamlit Web UI

```python
import streamlit as st
import cv2
import tempfile
from detector import detect_incidents
from alert import classify_event, check_alert, play_alert_sound

# --- Page Config ---
st.set_page_config(page_title="CCTV Incident Detector", layout="wide")
st.title("🎥 CCTV-Based Crime & Accident Detection System")
st.markdown("**Real-time incident analytics powered by YOLOv8**")

# --- Sidebar Settings ---
st.sidebar.header("⚙️ Settings")
source = st.sidebar.radio("Input Source", ["Upload Video", "Webcam"])

# --- Alert Log ---
alert_log = []

# --- Main Layout ---
col1, col2 = st.columns([3, 1])
with col1:
    video_placeholder = st.empty()
with col2:
    st.subheader("📋 Alert Log")
    log_placeholder = st.empty()

# --- Upload Video Mode ---
if source == "Upload Video":
    uploaded = st.sidebar.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if uploaded:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.read())
        cap = cv2.VideoCapture(tfile.name)
        st.sidebar.success("✅ Video loaded!")
        run = st.button("▶️ Start Detection")

        if run:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                detections, annotated = detect_incidents(frame)
                event = classify_event(detections)

                if event and check_alert(event):
                    alert_log.append(event)
                    play_alert_sound()

                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                video_placeholder.image(annotated_rgb, channels="RGB", use_column_width=True)
                log_placeholder.markdown("\n\n".join(f"🔴 {a}" for a in alert_log[-10:]))

            cap.release()
            st.success("✅ Detection Complete!")

# --- Webcam Mode ---
elif source == "Webcam":
    run = st.button("▶️ Start Webcam")
    if run:
        cap = cv2.VideoCapture(0)
        stop = st.button("⏹️ Stop")

        while cap.isOpened() and not stop:
            ret, frame = cap.read()
            if not ret:
                break

            detections, annotated = detect_incidents(frame)
            event = classify_event(detections)

            if event and check_alert(event):
                alert_log.append(event)
                play_alert_sound()

            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            video_placeholder.image(annotated_rgb, channels="RGB", use_column_width=True)
            log_placeholder.markdown("\n\n".join(f"🔴 {a}" for a in alert_log[-10:]))

        cap.release()
```

---

### `requirements.txt`

```
ultralytics
opencv-python-headless
streamlit
numpy
pillow
torch
torchvision
```

---

## 🔍 How Detection Works

### Stage 1 — Object Detection (YOLOv8)
- Each video frame is passed to the YOLOv8 nano model
- The model detects all objects with their class labels and confidence scores
- Only detections above **40% confidence** are kept

### Stage 2 — Event Classification
Rule-based logic classifies events:

| Detected Object | Event Raised |
|---|---|
| `knife` | ⚠️ CRIME DETECTED: Weapon Spotted |
| 4+ `person` | ⚠️ CROWD INCIDENT: Multiple Persons |
| `car` / `truck` / `bus` | 🚗 VEHICLE DETECTED: Possible Accident Zone |

### Stage 3 — Alert System
- 10-second cooldown prevents repeated alerts for same event
- All alerts logged with timestamp in the sidebar
- Alert log shows last 10 incidents

---

## ☁️ Cloud Deployment

This app is deployed on **Streamlit Cloud** (free tier):

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file: `app.py`
5. Click Deploy!

> ⚠️ Use `opencv-python-headless` in `requirements.txt` for cloud — NOT `opencv-python`

---

## 📊 Event Classification Summary

```
Input Video Frame
      │
      ▼
YOLOv8 Detection
      │
      ├── knife detected?          → 🔴 CRIME ALERT
      │
      ├── 4+ persons detected?     → 🟡 CROWD ALERT
      │
      ├── car/truck/bus detected?  → 🔵 VEHICLE ALERT
      │
      └── nothing suspicious?      → ✅ No Alert
```

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — Object Detection Model
- [Streamlit](https://streamlit.io) — Web UI Framework
- [OpenCV](https://opencv.org) — Video Processing
- Original GitHub Reference: [HasimRizvi/CCTVAccidentDetection](https://github.com/HasimRizvi/CCTVAccidentDetection)

---

## 👨‍💻 Author

**HasimRizvi**  
🌐 [Live App](https://cctv-final-project-kj6cdx6imicabgjjgc6vys.streamlit.app) | 💾 [GitHub Repo](https://github.com/HasimRizvi/cctv-final-project)

---

*Built with ❤️ using Python, YOLOv8, and Streamlit*

