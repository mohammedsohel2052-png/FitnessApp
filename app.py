import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import time

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="AI Fitness Gate", layout="wide")
st.title("🏋️ AI Fitness Gatekeeper")

# --- 2. AI MODEL SETUP (Outside the loop) ---
# We use @st.cache_resource so the "brain" file loads only once.
@st.cache_resource
def initialize_landmarker():
    model_path = 'pose_landmarker_lite.task' # Ensure this file is on GitHub
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO
    )
    return vision.PoseLandmarker.create_from_options(options)

landmarker = initialize_landmarker()

# --- 3. SIDEBAR / SETTINGS ---
with st.sidebar:
    st.header("Settings")
    injury = st.selectbox("Select Injury Area:", ["None", "Knee", "Shoulder"])
    target_reps = st.slider("Target Reps:", 1, 10, 5)

# --- 4. THE CAMERA LOOP ---
st.subheader("Complete your workout to unlock social media!")
frame_placeholder = st.empty() # Placeholder for the video feed
stop_button = st.button("Stop Workout")

# Use OpenCV to capture video
cap = cv2.VideoCapture(0)

while cap.isOpened() and not stop_button:
    success, frame = cap.read()
    if not success:
        st.error("Camera not detected.")
        break

    # Convert frame for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Run Detection (Requires timestamp for VIDEO mode)
    timestamp = int(time.time() * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp)

    # --- 5. YOUR FITNESS LOGIC GOES HERE ---
    if result.pose_landmarks:
        # Check coordinates and count reps
        # Example: st.write(f"Shoulder Y: {result.pose_landmarks[0][11].y}")
        pass

    # Display the live feed
    frame_placeholder.image(rgb_frame, channels="RGB")

cap.release()


