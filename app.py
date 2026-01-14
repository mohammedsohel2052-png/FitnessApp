import streamlit as st
import cv2
import mediapipe as mp

# --- INITIALIZATION ---
st.set_page_config(page_title="FitGate AI", page_icon="💪")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

if 'count' not in st.session_state:
    st.session_state.count = 0
    st.session_state.stage = None # "up" or "down"
    st.session_state.unlocked = False

# --- SIDEBAR: INJURY SELECTION ---
st.sidebar.title("Configuration")
user_injury = st.sidebar.selectbox("Injury Type:", ["None", "Knee", "Shoulder"])

# --- MAIN LOGIC ---
if not st.session_state.unlocked:
    st.title("🔒 Access Locked")
    st.subheader(f"Complete 5 Pushups to unlock Social Media")
    st.write(f"**Current Reps:** {st.session_state.count} / 5")

    # START CAMERA
    run = st.checkbox('Start Camera to Unlock')
    FRAME_WINDOW = st.image([])
    cam = cv2.VideoCapture(0)

    while run and st.session_state.count < 5:
        ret, frame = cam.read()
        if not ret: break
        
        # Convert to RGB for MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            # Get specific joints (Landmarks)
            landmarks = results.pose_landmarks.landmark
            shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
            elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y

            # Simple logic: If shoulder is below elbow = DOWN
            if shoulder > elbow:
                st.session_state.stage = "down"
            if shoulder < elbow and st.session_state.stage == "down":
                st.session_state.stage = "up"
                st.session_state.count += 1
                st.rerun() # Refresh UI to show new count

        FRAME_WINDOW.image(image)
    
    if st.session_state.count >= 5:
        st.session_state.unlocked = True
        cam.release()
        st.rerun()

else:
    # --- UNLOCKED CONTENT ---
    st.balloons()
    st.title("✅ Dashboard Unlocked")
    
    st.link_button("🚀 Open Instagram", "https://instagram.com")
    
    st.divider()
    st.header(f"Personalized Guidance: {user_injury}")
    
    if user_injury == "Knee":
        st.error("🛑 AVOID: Squats, Lunges")
        st.success("✅ DO: Wall Slides, Straight Leg Raises")
    elif user_injury == "Shoulder":
        st.error("🛑 AVOID: Pushups, Overhead Press")
        st.success("✅ DO: Pendulum Swings, Scapular Squeezes")
    else:
        st.success("✅ All exercises are safe for you!")
