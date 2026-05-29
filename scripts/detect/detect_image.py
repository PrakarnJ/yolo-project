import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


parser = argparse.ArgumentParser(description="Run YOLO detection on an image")
parser.add_argument("image", nargs="?", default=str(ROOT / "data/images/safety1.jpg"), help="Path to input image")
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

results = model(args.image, conf=conf)
r = results[0]

for box in r.boxes:
    cls_id = int(box.cls)
    conf_score = float(box.conf)
    xyxy = box.xyxy[0].tolist()
    print(f"{model.names[cls_id]}: {conf_score:.2f} | box: {[round(x) for x in xyxy]}")

output = ROOT / "runs/detect_image_output.jpg"
r.save(str(output))
print(f"Saved to {output}")
