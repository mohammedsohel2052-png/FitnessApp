import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2

# --- 1. SETUP ---
st.set_page_config(page_title="FitGate AI", page_icon="💪")
st.title("💪 FitGate AI: Vision-Based Workout")

if 'count' not in st.session_state:
    st.session_state.count = 0
if 'stage' not in st.session_state:
    st.session_state.stage = "up"

# --- 2. AI MODEL LOADING ---
# Ensure pose_landmarker_lite.task is in your GitHub!
model_path = 'pose_landmarker_lite.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.PoseLandmarker.create_from_options(options)

# --- 3. VIDEO PROCESSING LOGIC ---
class WorkoutProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Convert for MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        timestamp = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        
        # Detect landmarks
        result = detector.detect_for_video(mp_image, timestamp)
        
        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            # Use Shoulder Y (Landmark 11/12) for Pushup Logic
            shoulder_y = (landmarks[11].y + landmarks[12].y) / 2
            
            # Simple Counting Logic
            if shoulder_y > 0.6: # 'Down' position
                st.session_state.stage = "down"
            if shoulder_y < 0.4 and st.session_state.stage == "down": # 'Up' position
                st.session_state.stage = "up"
                st.session_state.count += 1

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 4. THE LIVE CAMERA GATE ---
webrtc_streamer(key="workout", video_processor_factory=WorkoutProcessor)

st.metric("Reps Completed", st.session_state.count)

if st.session_state.count >= 5:
    st.success("🎯 Goal Reached! Social Media Unlocked.")
    st.link_button("Open Instagram", "https://instagram.com")
