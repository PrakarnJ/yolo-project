import argparse
from collections import defaultdict
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def collect_images(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in SUPPORTED)


def run_model(entry, images, defaults):
    name = entry["name"]
    conf = entry.get("conf", defaults.get("conf", 0.5))
    imgsz = entry.get("imgsz", defaults.get("imgsz", 640))
    output_dir = resolve(defaults.get("project", "runs/detect")) / name
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(resolve(entry["weights"])))
    model.to("cuda")

    # per-class stats: {class_name: {"count": int, "conf_sum": float, "images": set}}
    stats = defaultdict(lambda: {"count": 0, "conf_sum": 0.0, "images": set()})
    total_detections = 0

    print(f"\n[{name}] Running on {len(images)} images ...")
    for img_path in images:
        results = model(str(img_path), conf=conf, imgsz=imgsz, verbose=False)
        r = results[0]

        for box in r.boxes:
            cls_name = model.names[int(box.cls)]
            conf_score = float(box.conf)
            stats[cls_name]["count"] += 1
            stats[cls_name]["conf_sum"] += conf_score
            stats[cls_name]["images"].add(img_path.name)
            total_detections += 1

        out_path = output_dir / img_path.name
        r.save(str(out_path))

    return {
        "name": name,
        "total": total_detections,
        "images_with_detections": len({img for s in stats.values() for img in s["images"]}),
        "stats": {
            cls: {
                "count": s["count"],
                "avg_conf": round(s["conf_sum"] / s["count"], 3),
                "in_images": len(s["images"]),
            }
            for cls, s in stats.items()
        },
        "output_dir": str(output_dir),
    }


def print_summary(all_results, num_images):
    all_classes = sorted({cls for r in all_results for cls in r["stats"]})
    model_names = [r["name"] for r in all_results]

    # Header
    print("\n" + "=" * 70)
    print("  DETECTION COMPARISON SUMMARY")
    print(f"  Images tested: {num_images}")
    print("=" * 70)

    # Overall totals
    name_w = max(len(n) for n in model_names)
    print(f"\n{'Model':<{name_w}}  {'Total Det':>10}  {'Images w/ Det':>14}")
    print("-" * (name_w + 30))
    for r in all_results:
        print(f"{r['name']:<{name_w}}  {r['total']:>10}  {r['images_with_detections']:>13}/{num_images}")

    # Per-class breakdown
    if all_classes:
        print(f"\n{'Class':<20}", end="")
        for r in all_results:
            print(f"  {r['name'][:18]:>18}", end="")
        print()
        print(f"{'':20}", end="")
        for _ in all_results:
            print(f"  {'cnt  avg_conf imgs':>18}", end="")
        print()
        print("-" * (20 + len(all_results) * 20))

        for cls in all_classes:
            print(f"{cls:<20}", end="")
            for r in all_results:
                s = r["stats"].get(cls)
                if s:
                    cell = f"{s['count']}  {s['avg_conf']:.3f}  {s['in_images']}"
                else:
                    cell = "-"
                print(f"  {cell:>18}", end="")
            print()
    else:
        print("\n  No detections across all models.")

    print("=" * 70)
    print("\nAnnotated outputs saved to:")
    for r in all_results:
        print(f"  {r['output_dir']}")


parser = argparse.ArgumentParser(description="Compare YOLO model detections on an image folder")
parser.add_argument("--config", required=True, help="Path to compare-detect config YAML")
args = parser.parse_args()

with open(resolve(args.config)) as f:
    cfg = yaml.safe_load(f)

images = collect_images(resolve(cfg["images"]))
if not images:
    raise FileNotFoundError(f"No images found in: {cfg['images']}")

print(f"Found {len(images)} images in {cfg['images']}")

all_results = []
for entry in cfg["models"]:
    all_results.append(run_model(entry, images, cfg))

print_summary(all_results, len(images))
