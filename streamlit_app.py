import streamlit as st
from ultralytics import YOLO
import cv2

st.set_page_config(
    page_title="HSE VisionGuard",
    page_icon="🦺",
    layout="wide"
)

st.title("🦺 HSE VisionGuard - Real-time Detection")

# Load model once
@st.cache_resource
def load_model():
    return YOLO(r"C:\Users\ADMIN\Desktop\Projects\HSE_VisionGuard\runs\detect\train2\weights\best.pt")

model = load_model()

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.05)
    camera_idx = st.selectbox("Camera", [0, 1, 2], index=0)
    st.markdown("---")
    st.markdown("**Classes:** Helmet, Vest")

# Center column
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    run = st.checkbox("▶️ Start Camera", value=False)
    FRAME_WINDOW = st.empty()
    st.caption("🎥 Real-time HSE Detection (Helmet & Vest)")

if run:
    cap = cv2.VideoCapture(camera_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to read from camera")
            break
        
        # Run YOLO inference
        results = model.predict(frame, conf=conf_threshold, verbose=False)
        annotated = results[0].plot()
        
        # Convert BGR to RGB and display
        FRAME_WINDOW.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    
    cap.release()
