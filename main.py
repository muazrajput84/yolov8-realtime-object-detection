"""
YOLOv8 Advanced Real-Time Object Detection
Author: Muaz
Version: 2.0 (Enhanced for Accuracy + Performance)

Features:
✅ Real-time webcam object detection
✅ Auto GPU/CPU detection
✅ FPS & Object Count display
✅ High-confidence filtering
✅ Auto model handling
✅ Screenshot saving (press 'S')
✅ Clean exit (press 'Q')
"""

import cv2
import os
import time
import torch
from datetime import datetime
from ultralytics import YOLO

# ---------------- SETTINGS ---------------- #
MODEL_SIZE = "yolov8m.pt"     # Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt
CONFIDENCE_THRESHOLD = 0.6     # Filter out low-confidence detections
SAVE_PATH = "screenshots"      # Folder to save captured frames
WINDOW_NAME = "YOLOv8 - Real-Time Detection"
# ------------------------------------------- #

# Ensure screenshot folder exists
os.makedirs(SAVE_PATH, exist_ok=True)

# Load YOLOv8 model with error handling
try:
    print(f"🔄 Loading YOLOv8 model: {MODEL_SIZE}")
    model = YOLO(MODEL_SIZE)
except Exception as e:
    print(f"⚠️ Model load failed: {e}")
    print("➡️ Downloading default model yolov8n.pt instead...")
    model = YOLO("yolov8n.pt")

# Use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
print(f"✅ Model loaded on: {device.upper()}")

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not access webcam.")
    exit()

print("\n🎥 Webcam started successfully!")
print("Press 'S' to save a screenshot | 'Q' to quit\n")

prev_time = 0  # For FPS calculation

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to capture frame.")
        break

    # Run detection
    results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
    annotated_frame = results[0].plot()

    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    # Count detected objects
    object_count = len(results[0].boxes)

    # Display info overlay
    cv2.putText(annotated_frame, f"FPS: {int(fps)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"Objects: {object_count}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    cv2.putText(annotated_frame, f"Model: {MODEL_SIZE}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Show window
    cv2.imshow(WINDOW_NAME, annotated_frame)

    # Key handling
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("👋 Exiting detection...")
        break
    elif key == ord('s'):
        filename = os.path.join(SAVE_PATH, f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        cv2.imwrite(filename, annotated_frame)
        print(f"💾 Screenshot saved: {filename}")

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("✅ Webcam released. Program terminated successfully.")
