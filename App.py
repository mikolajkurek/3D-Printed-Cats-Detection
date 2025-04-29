import cv2
import numpy as np
from ultralytics import YOLO
import time
from gpiozero import AngularServo

# Configuration
MODEL_PATH = "my_model.pt"  # Path to YOLO model
SOURCE = 0  # Source: 0 for USB camera, or path to video file (e.g., "testvid.mp4")
CONF_THRESH = 0.5  # Confidence threshold for detections

# Servo setup
servo = AngularServo(14, min_angle=0, max_angle=180, min_pulse_width=0.0005, max_pulse_width=0.0025)

# Load model
model = YOLO(MODEL_PATH, task='detect')
labels = model.names

# Initialize video capture
cap = cv2.VideoCapture(SOURCE)
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

# Colors for bounding boxes (RGB)
colors = [(164, 120, 87), (68, 148, 228), (93, 97, 209), (178, 182, 133), (88, 159, 106)]

# FPS tracking
fps_buffer = []
fps_avg_len = 50
avg_fps = 0

# Main loop
while True:
    t_start = time.perf_counter()

    # Read frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Run inference
    results = model(frame, verbose=False)
    detections = results[0].boxes
    object_count = 0
    
    servo.angle = 0

    # Draw detections
    for det in detections:
        xyxy = det.xyxy.cpu().numpy().squeeze().astype(int)
        conf = det.conf.item()
        if conf < CONF_THRESH:
            continue

        class_idx = int(det.cls.item())

        if labels[class_idx] == 'red_cat':
            print('Angle = 60')
            servo.angle = 60
        elif labels[class_idx] == 'yellow_cat':
            print('Angle = 120')
            servo.angle = 120
        elif labels[class_idx] == 'green_cat':
            print('Angle = 180')
            servo.angle = 180

        color = colors[class_idx % len(colors)]
        cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
        label = f'{labels[class_idx]}: {int(conf * 100)}%'
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        label_y = max(xyxy[1], label_size[1] + 10)
        cv2.rectangle(frame,
                      (xyxy[0], label_y - label_size[1] - 10),
                      (xyxy[0] + label_size[0], label_y + baseline - 10),
                      color, cv2.FILLED)
        cv2.putText(frame, label, (xyxy[0], label_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        object_count += 1

    # Display FPS and object count
    cv2.putText(frame, f'FPS: {avg_fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f'Objects: {object_count}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow('YOLO Detection', frame)

    # Handle keypress
    key = cv2.waitKey(5)
    if key == ord('q'):
        break
    if key == ord('s'):
        cv2.waitKey(0)
    if key == ord('p'):
        cv2.imwrite('capture.png', frame)

    # Calculate FPS
    fps = 1 / (time.perf_counter() - t_start)
    fps_buffer.append(fps)
    if len(fps_buffer) > fps_avg_len:
        fps_buffer.pop(0)
    avg_fps = np.mean(fps_buffer)

# Cleanup
print(f'Average FPS: {avg_fps:.2f}')
cap.release()
cv2.destroyAllWindows()
