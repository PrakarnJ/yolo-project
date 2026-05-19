from ultralytics import YOLO
import cv2
import sys

MODEL_PATH = "models/yolov8n.pt"
IMAGE_PATH = sys.argv[1] if len(sys.argv) > 1 else "data/images/harrypotter.jpg"

model = YOLO(MODEL_PATH)
model.to("cuda")

results = model(IMAGE_PATH, conf=0.5)
r = results[0]

# Print detections
for box in r.boxes:
    cls_id = int(box.cls)
    conf = float(box.conf)
    xyxy = box.xyxy[0].tolist()
    print(f"{model.names[cls_id]}: {conf:.2f} | box: {[round(x) for x in xyxy]}")

# Save output
r.save("runs/detect_image_output.jpg")
print("Saved to runs/detect_image_output.jpg")

