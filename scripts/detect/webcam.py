import argparse
import yaml
from pathlib import Path
import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


parser = argparse.ArgumentParser(description="Run YOLO detection from webcam")
parser.add_argument("--config", help="Path to detect config YAML (e.g. configs/detect_ppe_stecon.yaml)")
parser.add_argument("--model",  help="Path to model weights (overrides config)")
parser.add_argument("--conf",   type=float, help="Confidence threshold (overrides config)")
args = parser.parse_args()

cfg = {}
if args.config:
    with open(resolve(args.config)) as f:
        cfg = yaml.safe_load(f)

if args.model is not None:
    cfg["model"] = args.model
if args.conf is not None:
    cfg["conf"] = args.conf

model_path = str(resolve(cfg.get("model", "models/yolov8n.pt")))
conf = cfg.get("conf", 0.5)

model = YOLO(model_path)
model.to("cuda")

cap = cv2.VideoCapture(0)
print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame, stream=True, device=0, verbose=False, conf=conf)
    for r in results:
        cv2.imshow("YOLO Webcam", r.plot())
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
