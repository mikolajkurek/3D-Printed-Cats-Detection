import cv2
import numpy as np
import torch
import time

# Configuration
MODEL_PATH = "my_model.torchscript"  # Path to TorchScript model
SOURCE = 0  # Source: 0 for USB camera, or path to video file (e.g., "testvid.mp4")
CONF_THRESH = 0.5  # Confidence threshold for detections
IMG_SIZE = 640  # Input size expected by the model
IOU_THRESH = 0.45  # IoU threshold for NMS
LABELS = ['red_cat', 'yellow_cat', 'green_cat']  # Class labels (modify as per your model)

# Load TorchScript model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.jit.load(MODEL_PATH, map_location=device)
model.eval()
model.to(device)

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


# Image preprocessing function
def preprocess_image(frame, img_size=IMG_SIZE):
    # Resize image to model input size
    img = cv2.resize(frame, (img_size, img_size))
    # Convert to RGB (OpenCV uses BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Convert to tensor and normalize
    img = img.transpose((2, 0, 1)).astype(np.float32) / 255.0  # HWC to CHW
    img = torch.tensor(img, dtype=torch.float32)
    # Normalize with ImageNet mean and std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = (img - mean) / std
    # Add batch dimension
    img = img.unsqueeze(0).to(device)
    return img


# Post-process detections (apply NMS and scale coordinates)
def postprocess_detections(detections, frame_shape, img_size=IMG_SIZE, conf_thresh=CONF_THRESH, iou_thresh=IOU_THRESH):
    if detections is None or len(detections) == 0:
        return []

    # Assuming detections is a tensor of shape [N, 6] (x1, y1, x2, y2, conf, class)
    boxes = detections[:, :4]
    scores = detections[:, 4]
    classes = detections[:, 5].int()

    # Scale coordinates back to original frame size
    orig_h, orig_w = frame_shape[:2]
    scale_x, scale_y = orig_w / img_size, orig_h / img_size
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y

    # Apply NMS
    indices = torch.ops.torchvision.nms(boxes, scores, iou_thresh)
    boxes = boxes[indices]
    scores = scores[indices]
    classes = classes[indices]

    # Filter by confidence
    mask = scores >= conf_thresh
    boxes = boxes[mask]
    scores = scores[mask]
    classes = classes[mask]

    return list(zip(boxes.cpu().numpy(), scores.cpu().numpy(), classes.cpu().numpy()))


# Main loop
while True:
    t_start = time.perf_counter()

    # Read frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Preprocess frame
    input_tensor = preprocess_image(frame)

    # Run inference
    with torch.no_grad():
        outputs = model(input_tensor)

    # Parse model output (adjust based on your model's output format)
    # Example: assuming outputs is a tensor [N, 6] (x1, y1, x2, y2, conf, class)
    detections = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
    detections = postprocess_detections(detections, frame.shape)

    object_count = 0

    # Draw detections
    for box, conf, class_idx in detections:
        xyxy = box.astype(int)
        class_name = LABELS[class_idx]

        if class_name == 'red_cat':
            print('left')
        elif class_name == 'yellow_cat':
            print('right')
        elif class_name == 'green_cat':
            print('straight')

        color = colors[class_idx % len(colors)]
        cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
        label = f'{class_name}: {int(conf * 100)}%'
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
    cv2.imshow('TorchScript Detection', frame)

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