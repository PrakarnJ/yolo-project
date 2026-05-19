from ultralytics import YOLO
import cv2

MODEL_PATH = "models/yolov8n.pt"

model = YOLO(MODEL_PATH)
model.to("cuda")

cap = cv2.VideoCapture(0)

print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True, device=0, verbose=False, conf=0.5)

    for r in results:
        annotated = r.plot()
        cv2.imshow("YOLO Webcam", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
