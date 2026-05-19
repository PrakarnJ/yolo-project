from ultralytics import YOLO
import cv2
import sys

MODEL_PATH = "models/yolov8n.pt"
VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else "data/videos/test.mp4"
OUTPUT_PATH = "runs/detect_video_output.mp4"

model = YOLO(MODEL_PATH)
model.to("cuda")

cap = cv2.VideoCapture(VIDEO_PATH)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, device=0, verbose=False, conf=0.5)
    annotated = results[0].plot()

    out.write(annotated)
    frame_count += 1
    print(f"Frame {frame_count}", end="\r")

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"\nDone. Saved to {OUTPUT_PATH}")
