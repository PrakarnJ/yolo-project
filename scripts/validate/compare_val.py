import argparse
import csv
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]

METRICS = ["mAP50", "mAP50-95", "Precision", "Recall"]


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def run_val(entry, defaults):
    weights = entry["weights"]
    name = entry["name"]
    model = YOLO(str(resolve(weights)))
    metrics = model.val(
        data=str(resolve(entry.get("dataset", defaults["dataset"]))),
        split=entry.get("split", defaults.get("split", "val")),
        imgsz=entry.get("imgsz", defaults.get("imgsz", 640)),
        batch=entry.get("batch", defaults.get("batch", 16)),
        device=0,
        conf=entry.get("conf", defaults.get("conf", 0.5)),
        iou=entry.get("iou", defaults.get("iou", 0.5)),
        plots=True,
        save_json=False,
        project=str(resolve(defaults.get("project", "runs/val"))),
        name=name,
        verbose=False,
    )
    return {
        "Model":     name,
        "Weights":   weights,
        "mAP50":     round(metrics.box.map50, 4),
        "mAP50-95":  round(metrics.box.map,   4),
        "Precision": round(metrics.box.mp,    4),
        "Recall":    round(metrics.box.mr,    4),
    }


def print_table(rows):
    headers = ["Model", "Weights", "mAP50", "mAP50-95", "Precision", "Recall"]
    widths = {h: max(len(h), max(len(str(r[h])) for r in rows)) for h in headers}
    sep = "  ".join("-" * widths[h] for h in headers)
    fmt = lambda r: "  ".join(str(r[h]).ljust(widths[h]) for h in headers)  # noqa: E731

    print("\n" + "=" * len(sep))
    print("  MODEL COMPARISON SUMMARY")
    print("=" * len(sep))
    print(fmt({h: h for h in headers}))
    print(sep)

    # highlight best metric in each column
    best = {m: max(r[m] for r in rows) for m in METRICS}
    for r in rows:
        row_str = []
        for h in headers:
            cell = str(r[h]).ljust(widths[h])
            if h in METRICS and r[h] == best[h]:
                cell = cell + " *"
            row_str.append(cell)
        print("  ".join(row_str))

    print(sep)
    print("* = best in column")


def save_csv(rows, path):
    headers = ["Model", "Weights", "mAP50", "mAP50-95", "Precision", "Recall"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


parser = argparse.ArgumentParser(description="Compare multiple YOLO model weights on a dataset")
parser.add_argument("--config", required=True, help="Path to compare config YAML")
parser.add_argument("--output", help="Override: path to save CSV (default: runs/val/comparison.csv)")
args = parser.parse_args()

with open(resolve(args.config)) as f:
    cfg = yaml.safe_load(f)

if "dataset" not in cfg:
    raise ValueError("Config must have a top-level 'dataset' key")
if "models" not in cfg or not cfg["models"]:
    raise ValueError("Config must have a non-empty 'models' list")

results = []
total = len(cfg["models"])
for i, entry in enumerate(cfg["models"], 1):
    print(f"\n[{i}/{total}] Validating: {entry['name']} ({entry['weights']})")
    results.append(run_val(entry, cfg))

print_table(results)

output_path = resolve(args.output) if args.output else resolve(cfg.get("project", "runs/val")) / "comparison.csv"
save_csv(results, output_path)
print(f"\nCSV saved to {output_path}")
