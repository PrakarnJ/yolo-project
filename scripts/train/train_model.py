import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


parser = argparse.ArgumentParser(description="Train a YOLO model")
parser.add_argument("--config", required=True, help="Path to train config YAML (e.g. configs/train_ppe_stecon.yaml)")
parser.add_argument("--model",    help="Override: path to base weights")
parser.add_argument("--data",     help="Override: path to dataset YAML")
parser.add_argument("--epochs",   type=int,   help="Override: number of epochs")
parser.add_argument("--imgsz",    type=int,   help="Override: image size")
parser.add_argument("--batch",    type=int,   help="Override: batch size")
parser.add_argument("--freeze",   type=int,   help="Override: number of layers to freeze")
parser.add_argument("--lr0",      type=float, help="Override: initial learning rate")
parser.add_argument("--patience", type=int,   help="Override: early stopping patience")
parser.add_argument("--name",                 help="Override: run name")
args = parser.parse_args()

with open(resolve(args.config)) as f:
    cfg = yaml.safe_load(f)

# Apply CLI overrides
for key in ["model", "data", "epochs", "imgsz", "batch", "freeze", "lr0", "patience", "name"]:
    val = getattr(args, key)
    if val is not None:
        cfg[key] = val

model = YOLO(str(resolve(cfg["model"])))

model.train(
    data=str(resolve(cfg["data"])),
    epochs=cfg.get("epochs", 100),
    imgsz=cfg.get("imgsz", 640),
    batch=cfg.get("batch", 16),
    device=0,
    freeze=cfg.get("freeze", 0),
    lr0=cfg.get("lr0", 0.01),
    patience=cfg.get("patience", 50),
    project=str(resolve(cfg.get("project", "runs/train"))),
    name=cfg.get("name", "experiment"),
)
