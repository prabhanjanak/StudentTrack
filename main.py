import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Load YOLOv8 model
model_path = "best.pt"  # Update with your trained YOLOv8 model path
model = YOLO(model_path)

# Streamlit UI
st.title("ResoluteAI Software - Exam Cheating Detection")

# Upload Image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# Checkbox to detect boxes
detect_boxes = st.checkbox("Show detection boxes")

if uploaded_image:
    # Convert image to OpenCV format
    image = Image.open(uploaded_image)
    image = np.array(image)

    # Perform object detection
    results = model(image)

    # Get detected classes and confidence
    total_boxes = 0
    cheating_count = 0

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])  # Class ID
            conf = float(box.conf[0])  # Confidence
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            
            # Count cheating cases
            if cls == 1:  # Assuming class 1 = Cheating
                cheating_count += 1
            
            total_boxes += 1

            # Draw bounding boxes if checked
            if detect_boxes:
                color = (0, 255, 0) if cls == 0 else (0, 0, 255)
                label = "Not Cheating" if cls == 0 else "Cheating"
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Calculate cheating percentage
    cheating_percentage = (cheating_count / total_boxes) * 100 if total_boxes > 0 else 0

    # Display detected image
    st.image(image, caption="Detected Image", use_column_width=True)

    # Show cheating percentage
    st.write(f"### Cheating Probability: {cheating_percentage:.2f}%")
