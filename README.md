# Classroom Exam Cheating Detection System

## Overview
This project implements an AI-based **cheating detection system** for classroom exams using **video analysis**. The system detects and tracks students, analyzes their posture, and flags suspicious behaviors using **YOLOv8, DeepSORT, MediaPipe/OpenPose, LSTM + CNN, and OpenCV**.

## Features
- **Student Detection**: Uses YOLOv8 to detect students in the exam hall.
- **Tracking**: Implements DeepSORT for tracking students across frames.
- **Pose Estimation**: Uses MediaPipe to extract pose keypoints.
- **Behavior Classification**: LSTM + CNN classify student activities as cheating or not.
- **Alert System**: Flags suspicious behavior and highlights students with red bounding boxes.

## Installation
### 1. Setup Google Colab Environment
```bash
!pip install ultralytics opencv-python mediapipe torch torchvision deep-sort-realtime
!git clone https://github.com/mikel-brostrom/Yolov8_DeepSORT_OSNet.git
```

### 2. Mount Google Drive & Extract Dataset
```python
from google.colab import drive
import zipfile

drive.mount('/content/drive')
dataset_zip_path = "/content/drive/MyDrive/dataset.zip"
extract_path = "/content/dataset"

with zipfile.ZipFile(dataset_zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
print("Dataset extracted successfully.")
```

## Training YOLOv8 for Student Detection
```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(data="/content/dataset/data.yaml", epochs=50, batch=16, imgsz=640)
```

## Running Inference
```python
model = YOLO("/content/runs/detect/train/weights/best.pt")
model.predict(source="/content/test_video.mp4", save=True, conf=0.5)
```

## Tracking with DeepSORT
```bash
cd Yolov8_DeepSORT_OSNet
python track.py --yolo_model "/content/runs/detect/train/weights/best.pt" --source "/content/test_video.mp4" --output "/content/output.mp4"
```

## Pose Estimation with MediaPipe
```python
import cv2
import mediapipe as mp
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
cap = cv2.VideoCapture("/content/test_video.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.imshow('Pose Detection', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

## Behavior Classification with LSTM + CNN
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(30, 33)),
    Dropout(0.2),
    LSTM(64),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

## Alert System for Cheating Detection
```python
import cv2

def draw_alert(frame, bbox, text="Cheating Detected"):
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
```

## Future Improvements
- Enhance dataset for improved behavior classification.
- Deploy real-time monitoring on edge devices.
- Add audio-based analysis for voice cheating detection.

## License
MIT License

---
This **README.md** provides a complete guide to setting up and running the project. 🚀

