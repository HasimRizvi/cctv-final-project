import streamlit as st
import cv2
import tempfile
from detector import detect_incidents
from alert import classify_event, check_alert, play_alert_sound

st.set_page_config(page_title="CCTV Incident Detector", layout="wide")
st.title("🎥 CCTV-Based Crime & Accident Detection System")
st.markdown("**Real-time incident analytics powered by YOLOv8**")

st.sidebar.header("⚙️ Settings")
source = st.sidebar.radio("Input Source", ["Upload Video", "Webcam"])

alert_log = []

col1, col2 = st.columns([3, 1])

with col1:
    video_placeholder = st.empty()

with col2:
    st.subheader("📋 Alert Log")
    log_placeholder = st.empty()

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