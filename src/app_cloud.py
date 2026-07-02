"""
EdgeGesture-HCI — Cloud Deployment Version
Uses streamlit-webrtc for browser-based webcam streaming.
Works on Streamlit Community Cloud, Hugging Face Spaces, etc.
"""

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import numpy as np
import os
import sys
import threading
from collections import deque

# ─── Path Setup ───────────────────────────────────────────────
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, BASE_DIR)

from config import GESTURE_LABELS, GESTURE_TO_COMMAND, COLORS, MEDIAPIPE_CONFIG
from hand_tracker import HandTracker
from gesture_classifier import GestureClassifier

# ─── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="EdgeGesture-HCI | Space Mission Gesture Control",
    page_icon="🚀",
    layout="wide",
)

# ─── Load Model (cached) ─────────────────────────────────────
@st.cache_resource
def load_classifier():
    model_path = os.path.join(BASE_DIR, "models", "gesture_model_robust.h5")
    encoder_path = os.path.join(BASE_DIR, "models", "label_encoder.pkl")
    return GestureClassifier(model_path=model_path, encoder_path=encoder_path)

classifier = load_classifier()


# ─── WebRTC Video Processor ──────────────────────────────────
class GestureProcessor(VideoProcessorBase):
    """Processes each video frame: detect hand → classify gesture → draw overlay."""

    def __init__(self):
        # Each WebRTC session gets its own tracker (thread-safe)
        self.tracker = HandTracker(
            max_hands=MEDIAPIPE_CONFIG["max_num_hands"],
            detection_conf=MEDIAPIPE_CONFIG["min_detection_confidence"],
            tracking_conf=MEDIAPIPE_CONFIG["min_tracking_confidence"],
        )
        self.gesture = "background"
        self.command = "NO COMMAND (idle)"
        self.confidence = 0.0
        self.finger_states = None
        self.lock = threading.Lock()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Process a single video frame from the browser webcam."""
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # Mirror effect

        # Step 1: Detect hand landmarks
        results = self.tracker.process_frame(img)
        landmarks = self.tracker.get_landmark_array(results)

        # Step 2: Classify gesture
        classifier.update(landmarks)
        gesture = classifier.classify(landmarks)
        command = GESTURE_TO_COMMAND.get(gesture, "UNKNOWN")

        # Step 3: Get confidence
        confidence = 0.0
        if classifier.use_model and landmarks is not None:
            flat = landmarks.flatten().reshape(1, -1)
            pred = classifier.model.predict(flat, verbose=0)
            confidence = float(np.max(pred[0])) * 100

        # Step 4: Get finger states
        finger_states = None
        if landmarks is not None:
            finger_states = self.tracker.get_finger_states(landmarks)

        # Thread-safe update of shared state
        with self.lock:
            self.gesture = gesture
            self.command = command
            self.confidence = confidence
            self.finger_states = finger_states

        # Step 5: Draw landmarks on frame
        img = self.tracker.draw_landmarks(img, results)

        # Step 6: Draw text overlay
        color = (0, 255, 170) if gesture != "background" else (150, 150, 150)
        cv2.putText(img, f"Gesture: {gesture}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(img, f"Command: {command}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 200, 66), 2)
        cv2.putText(img, f"Confidence: {confidence:.1f}%", (10, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ─── Custom CSS ───────────────────────────────────────────────
st.markdown(f"""
<style>
    .main-title {{
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        color: {COLORS['accent_green']};
        margin-bottom: 0.5rem;
    }}
    .subtitle {{
        text-align: center;
        color: grey;
        margin-bottom: 1.5rem;
    }}
    .gesture-box {{
        text-align: center;
        padding: 20px;
        border-radius: 15px;
        background: linear-gradient(135deg, {COLORS['bg_secondary']}, {COLORS['accent_blue']});
        border: 2px solid {COLORS['accent_green']};
        margin: 10px 0;
    }}
    .gesture-label {{
        font-size: 2rem;
        font-weight: bold;
        color: {COLORS['accent_green']};
    }}
    .command-text {{
        font-size: 1.2rem;
        color: {COLORS['accent_yellow']};
    }}
    .status-idle {{
        color: {COLORS['text_secondary']};
        font-size: 1.2rem;
    }}
    .status-active {{
        color: {COLORS['accent_green']};
        animation: pulse 1s infinite;
    }}
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.5; }}
    }}
    .info-card {{
        padding: 15px;
        border-radius: 10px;
        background: {COLORS['bg_secondary']};
        border: 1px solid {COLORS['accent_blue']};
        margin: 8px 0;
    }}
</style>
""", unsafe_allow_html=True)

# ─── Title ────────────────────────────────────────────────────
st.markdown('<div class="main-title">🚀 EdgeGesture-HCI</div>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Real-Time Contactless Gesture Control for Space Missions | ISRO × MANIT Bhopal</p>', unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    st.divider()

    # Gesture reference table
    st.header("📋 Gesture Commands")
    for gesture, command in GESTURE_TO_COMMAND.items():
        if gesture != "background":
            st.text(f"  {gesture:12s} -> {command}")

    st.divider()
    st.caption(f"Model: CNN-GRU | Accuracy: 91.72%")
    st.caption(f"Mode: {'Deep Learning' if classifier.use_model else 'Rule-based'}")
    st.caption("MediaPipe: Tasks API v0.10+")

# ─── Main Layout ──────────────────────────────────────────────
col_cam, col_info = st.columns([2, 1])

with col_cam:
    st.subheader("📷 Live Camera Feed")
    st.caption("Allow camera access when prompted by your browser.")

    # WebRTC streamer — works on cloud!
    # STUN server helps establish peer connection through firewalls
    RTC_CONFIGURATION = {
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }

    ctx = webrtc_streamer(
        key="gesture-recognition",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=GestureProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col_info:
    st.subheader("🎯 Detection")
    gesture_placeholder = st.empty()
    command_placeholder = st.empty()
    finger_placeholder = st.empty()
    confidence_placeholder = st.empty()

# ─── Live Info Updates ────────────────────────────────────────
# Update the info panel while the stream is active
if ctx.video_processor:
    import time

    while ctx.state.playing:
        with ctx.video_processor.lock:
            gesture = ctx.video_processor.gesture
            command = ctx.video_processor.command
            confidence = ctx.video_processor.confidence
            finger_states = ctx.video_processor.finger_states

        # Update gesture display
        with col_info:
            if gesture != "background":
                gesture_placeholder.markdown(
                    f'<div class="gesture-box">'
                    f'<div class="gesture-label">{gesture.upper()}</div>'
                    f'<div class="command-text">{command}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                gesture_placeholder.markdown(
                    f'<div class="gesture-box">'
                    f'<div class="status-idle">IDLE — No gesture detected</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            # Show confidence
            confidence_placeholder.metric("Confidence", f"{confidence:.1f}%")

            # Show finger states
            if finger_states:
                fingers = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
                finger_text = " | ".join(
                    f"{fingers[i]}: {'UP' if finger_states[i] else 'DN'}"
                    for i in range(5)
                )
                finger_placeholder.caption(f"Fingers: {finger_text}")

        time.sleep(0.1)  # Update 10 times per second

# ─── Info Section (when camera is not active) ─────────────────
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="info-card">
    <h4>🧠 Model</h4>
    <p>CNN-GRU Hybrid Architecture<br>
    91.72% Test Accuracy<br>
    11 Gesture Classes<br>
    ~4MB Model Size</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
    <h4>🛡️ Robustness</h4>
    <p>Trained with 12x augmentation:<br>
    • Microgravity jitter<br>
    • Scale variation<br>
    • Fingertip occlusion</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-card">
    <h4>🚀 Space Mission Ready</h4>
    <p>Designed for ISRO missions:<br>
    • Contactless control<br>
    • Edge-deployable<br>
    • Real-time at 30 FPS</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()
st.caption("Built at MANIT Bhopal | ISRO-affiliated research project | CNN-GRU + MediaPipe")
