import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


parser = argparse.ArgumentParser(description="Validate a YOLO model")
parser.add_argument("--config", required=True, help="Path to val config YAML (e.g. configs/val_ppe_stecon.yaml)")
parser.add_argument("--model",  help="Override: path to model weights")
parser.add_argument("--data",   help="Override: path to dataset YAML")
parser.add_argument("--split",  choices=["train", "val", "test"], help="Override: dataset split")
parser.add_argument("--imgsz",  type=int,   help="Override: image size")
parser.add_argument("--batch",  type=int,   help="Override: batch size")
parser.add_argument("--conf",   type=float, help="Override: confidence threshold")
parser.add_argument("--iou",    type=float, help="Override: IoU threshold")
parser.add_argument("--name",               help="Override: run name")
args = parser.parse_args()

with open(resolve(args.config)) as f:
    cfg = yaml.safe_load(f)

# Apply CLI overrides
for key in ["model", "data", "split", "imgsz", "batch", "conf", "iou", "name"]:
    val = getattr(args, key)
    if val is not None:
        cfg[key] = val

model = YOLO(str(resolve(cfg["model"])))

metrics = model.val(
    data=str(resolve(cfg["data"])),
    split=cfg.get("split", "val"),
    imgsz=cfg.get("imgsz", 640),
    batch=cfg.get("batch", 16),
    device=0,
    conf=cfg.get("conf", 0.5),
    iou=cfg.get("iou", 0.5),
    plots=True,
    save_json=False,
    project=str(resolve(cfg.get("project", "runs/val"))),
    name=cfg.get("name", "experiment"),
)

print(f"\nmAP50:     {metrics.box.map50:.4f}")
print(f"mAP50-95:  {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall:    {metrics.box.mr:.4f}")
