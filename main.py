import streamlit as st
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image

# Set page title and layout
st.set_page_config(page_title="Exam Cheating Detection", layout="wide")

# Title
st.title("🎓 ResoluteAI - Exam Cheating Detection System")

# Set device (CUDA if available, else CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
st.write(f"🖥️ **Running on:** {device.upper()}")

# Load YOLOv8 model
MODEL_PATH = "best.pt"

try:
    model = YOLO(MODEL_PATH).to(device)
    st.success("✅ Model Loaded Successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# File uploader for image input
uploaded_image = st.file_uploader("📤 Upload an Image", type=["jpg", "png", "jpeg"])

# Checkbox for showing bounding boxes
detect_boxes = st.checkbox("Show detection boxes", value=True)

if uploaded_image:
    st.write("📊 **Processing Image...**")

    # Convert PIL image to OpenCV format (BGR)
    image = Image.open(uploaded_image)
    image_np = np.array(image)

    if image_np.shape[-1] == 4:  # Convert RGBA to RGB
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

    # Perform object detection
    results = model(image_np)

    # Initialize counters
    total_boxes, cheating_count = 0, 0

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])  # Class ID
            conf = round(float(box.conf[0]) * 100, 2)  # Confidence in %
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            
            # Count cheating cases
            if cls == 1:  # Assuming class 1 = Cheating
                cheating_count += 1
            
            total_boxes += 1

            # Draw bounding boxes if enabled
            if detect_boxes:
                color = (0, 255, 0) if cls == 0 else (0, 0, 255)  # Green for Not Cheating, Red for Cheating
                label = "Not Cheating" if cls == 0 else "Cheating"
                
                # Draw rectangle & label
                cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 3)
                cv2.putText(image_np, f"{label} ({conf}%)", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Compute cheating probability
    cheating_percentage = (cheating_count / total_boxes) * 100 if total_boxes > 0 else 0

    # Convert image back to RGB for display in Streamlit
    image_result = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    # Display the processed image
    st.image(image_result, caption="🖼️ Processed Image", use_column_width=True)

    # Show cheating probability
    st.subheader(f"📊 **Cheating Probability: {cheating_percentage:.2f}%**")
