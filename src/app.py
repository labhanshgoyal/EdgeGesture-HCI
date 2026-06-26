import streamlit as st
import cv2
import numpy as np
import os
import sys
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import GESTURE_LABELS, GESTURE_TO_COMMAND, COLORS, MEDIAPIPE_CONFIG
from hand_tracker import HandTracker
from gesture_classifier import GestureClassifier

from spacecraft_ui import SpacecraftUI
from metrics_dashboard import MetricsDashboard

st.set_page_config(page_title="Hand Gesture Recognition For Space Missions", layout="wide")

# ─── Load Model ────────────────────────────────────────────────────
@st.cache_resource
def load_classifier():
    """Load the gesture classifier (cached so it only loads once)."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "gesture_model_robust.h5")
    encoder_path = os.path.join(base_dir, "models", "label_encoder.pkl")
    return GestureClassifier(model_path=model_path, encoder_path=encoder_path)

@st.cache_resource
def load_tracker():
    """Load the hand tracker (cached)."""
    return HandTracker(
        max_hands=MEDIAPIPE_CONFIG["max_num_hands"],
        detection_conf=MEDIAPIPE_CONFIG["min_detection_confidence"],
        tracking_conf=MEDIAPIPE_CONFIG["min_tracking_confidence"],
    )

classifier = load_classifier()
tracker = load_tracker()
spacecraft = SpacecraftUI()
dashboard = MetricsDashboard()

# ─── Session State for live accuracy tracking ─────────────────
if "predictions" not in st.session_state:
    st.session_state.predictions = deque(maxlen=100)  # Track last 100 predictions
    st.session_state.correct_count = 0
    st.session_state.total_count = 0
    st.session_state.avg_confidence = 0.0

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
    }}
    .status-active {{
        color: {COLORS['accent_green']};
        animation: pulse 1s infinite;
    }}
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.5; }}
    }}
</style>
""", unsafe_allow_html=True)

# ─── Title ────────────────────────────────────────────────────
st.markdown('<div class="main-title">🚀 EdgeGesture</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:grey;">Real-Time Contactless Gesture Control</p>', unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    # Confidence threshold slider
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.3,
        max_value=0.95,
        value=0.6,
        step=0.05,
        help="Minimum confidence to accept a gesture"
    )

    # Camera selector
    camera_index = st.selectbox(
        "📷 Camera Source",
        options=[0, 1, 2],
        index=1,
        format_func=lambda x: f"Camera {x}" + (" (Built-in)" if x == 0 else " (USB)" if x == 1 else ""),
        help="0 = Built-in camera, 1 = USB webcam, 2 = Other"
    )

    st.divider()

    # Gesture reference table
    st.header("📋 Gesture Commands")
    for gesture, command in GESTURE_TO_COMMAND.items():
        if gesture != "background":
            st.text(f"  {gesture:12s} -> {command}")

    st.divider()
    live_acc = st.session_state.avg_confidence
    st.caption(f"Model: CNN+GRU | Live Confidence: {live_acc:.1f}%")
    st.caption(f"Predictions: {st.session_state.total_count}")
    st.caption(f"Mode: {'Model' if classifier.use_model else 'Rule-based'}")
    accuracy_placeholder = st.empty()

# ─── Main Layout ──────────────────────────────────────────────
col_cam, col_info = st.columns([2, 1])

with col_cam:
    st.subheader("📷 Live Camera Feed")
    camera_placeholder = st.empty()
    status_placeholder = st.empty()

with col_info:
    st.subheader("🎯 Detection")
    gesture_placeholder = st.empty()
    command_placeholder = st.empty()
    landmarks_placeholder = st.empty()

# ─── Webcam Loop ──────────────────────────────────────────────
start_button = st.button("▶️ Start Camera", width='stretch')
stop_button = st.button("⏹️ Stop Camera", width='stretch')

if start_button:
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        st.error("Could not open webcam! Check your camera connection.")
    else:
        status_placeholder.success("Camera running...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to read frame from camera.")
                break

            # Flip frame (mirror effect — more natural)
            frame = cv2.flip(frame, 1)

            # Step 1: Detect hand landmarks
            results = tracker.process_frame(frame)

            # Step 2: Extract landmark array
            landmarks = tracker.get_landmark_array(results)

            # Step 3: Update classifier buffer + classify
            classifier.update(landmarks)
            gesture = classifier.classify(landmarks)
            command = GESTURE_TO_COMMAND.get(gesture, "UNKNOWN")

            # Track live confidence
            if classifier.use_model and landmarks is not None:
                flat = landmarks.flatten().reshape(1, -1)
                pred = classifier.model.predict(flat, verbose=0)
                confidence = float(np.max(pred[0])) * 100
                st.session_state.predictions.append(confidence)
                st.session_state.total_count += 1
                st.session_state.avg_confidence = np.mean(list(st.session_state.predictions))

            # Step 4: Draw landmarks on frame
            frame = tracker.draw_landmarks(frame, results)

            # Step 5: Add gesture text on frame
            color = (0, 255, 170) if gesture != "background" else (150, 150, 150)
            cv2.putText(frame, f"Gesture: {gesture}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.putText(frame, f"Command: {command}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 200, 66), 2)

            # Step 6: Display in Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            camera_placeholder.image(frame_rgb, channels="RGB", width='stretch')

            # ─── Spacecraft Cockpit ──────────────────────
            st.markdown("---")
            spacecraft.render_all(gesture, command)
            # ─── Performance Dashboard ───────────────────
            st.markdown("---")
            dashboard.render_all(gesture, confidence if classifier.use_model else None)

            # Step 7: Update info panel
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
                        f'<div class="status-idle">IDLE — No gesture</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                # Show finger states
                if landmarks is not None:
                    finger_states = tracker.get_finger_states(landmarks)
                    if finger_states:
                        fingers = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
                        finger_text = " | ".join(
                            f"{fingers[i]}: {'UP' if finger_states[i] else 'DN'}"
                            for i in range(5)
                        )
                        landmarks_placeholder.caption(f"Fingers: {finger_text}")

            # Check if stop button pressed
            if stop_button:
                break

        cap.release()
        status_placeholder.info("Camera stopped.")