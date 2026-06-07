import argparse
import yaml
from pathlib import Path
import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


parser = argparse.ArgumentParser(description="Run YOLO detection on a video file")
parser.add_argument("video", nargs="?", default=str(ROOT / "data/videos/test.mp4"), help="Path to input video")
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

output = ROOT / "runs/detect_video_output.mp4"
cap = cv2.VideoCapture(args.video)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame, device=0, verbose=False, conf=conf)
    out.write(results[0].plot())
    frame_count += 1
    print(f"Frame {frame_count}", end="\r")

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"\nDone. Saved to {output}")
