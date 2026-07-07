"""
Blind test on an external folder of images against all trained models in a workspace.

Usage:
  # Visual only (no labels)
  .venv/bin/python scripts/blind_test.py --config configs/retrain/physical_fight.yaml \\
      --test-images data/my_test/images/

  # With YOLO-format labels → also computes mAP metrics for YOLO models
  .venv/bin/python scripts/blind_test.py --config configs/retrain/ppe.yaml \\
      --test-images data/my_test/images/ \\
      --labels      data/my_test/labels/

Suggested test folder layout:
  data/
  └── my_test/
      ├── images/   ← .jpg / .jpeg / .png files
      └── labels/   ← optional YOLO .txt files (same stem as images)

Output is written to:
  <workspace_dir>/external_blind_test_YYYYMMDD_HHMMSS/
      run_001_.../
          yolo/     ← annotated images from YOLO best.pt
          rfdetr/   ← annotated images from RF-DETR checkpoint
      summary.csv   ← per-run metrics (if --labels provided, otherwise "N/A")
"""

import argparse
import csv
import os
import shutil
import tempfile

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def _detect_yolo(run_dir: Path, test_imgs: list, out_dir: Path, val_cfg: dict):
    from ultralytics import YOLO

    weights = run_dir / "weights" / "best.pt"
    if not weights.exists():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    conf  = val_cfg.get("conf", 0.5)
    imgsz = val_cfg.get("imgsz", 640)
    for img_path in test_imgs:
        result = model(str(img_path), conf=conf, imgsz=imgsz, verbose=False)[0]
        result.save(str(out_dir / img_path.name))
    print(f"    → yolo/ ({len(test_imgs)} images)")


def _detect_rfdetr(run_dir: Path, test_imgs: list, out_dir: Path,
                   val_cfg: dict, class_names: list):
    import warnings
    import cv2
    import numpy as np
    import supervision as sv
    warnings.filterwarnings("ignore", category=FutureWarning, module="rfdetr")
    from rfdetr import RFDETR

    weights = run_dir / "rfdetr" / "checkpoint_best_total.pth"
    if not weights.exists():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    model           = RFDETR.from_checkpoint(str(weights))
    box_annotator   = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    conf            = val_cfg.get("conf", 0.5)
    for img_path in test_imgs:
        dets = model.predict(str(img_path), threshold=conf)
        img  = cv2.imread(str(img_path))
        if dets is not None and len(dets) > 0:
            sv_dets = sv.Detections(
                xyxy=np.array(dets.xyxy),
                confidence=np.array(dets.confidence),
                class_id=np.array(dets.class_id, dtype=int),
            )
            labels = [
                f"{class_names[int(c)] if int(c) < len(class_names) else c} {s:.2f}"
                for c, s in zip(dets.class_id, dets.confidence)
            ]
            img = box_annotator.annotate(scene=img, detections=sv_dets)
            img = label_annotator.annotate(scene=img, detections=sv_dets, labels=labels)
        cv2.imwrite(str(out_dir / img_path.name), img)
    print(f"    → rfdetr/ ({len(test_imgs)} images)")


def _eval_yolo_metrics(run_dir: Path, test_images_dir: Path, labels_dir: Path,
                       class_names: list, val_cfg: dict, tmp_root: Path) -> dict | None:
    from ultralytics import YOLO

    weights = run_dir / "weights" / "best.pt"
    if not weights.exists():
        return None

    tmp = tmp_root / run_dir.name
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "images").symlink_to(test_images_dir.resolve())
    (tmp / "labels").symlink_to(labels_dir.resolve())

    dataset_yaml = tmp / "dataset.yaml"
    dataset_yaml.write_text(yaml.dump({
        "path":  str(tmp),
        "train": "images",  # unused (split="test" below) but required by YOLO's schema check
        "val":   "images",  # unused (split="test" below) but required by YOLO's schema check
        "test":  "images",
        "names": {i: n for i, n in enumerate(class_names)},
        "nc":    len(class_names),
    }))

    try:
        metrics = YOLO(str(weights)).val(
            data=str(dataset_yaml),
            split="test",
            imgsz=val_cfg.get("imgsz", 640),
            batch=val_cfg.get("batch", 16),
            conf=val_cfg.get("conf", 0.5),
            iou=val_cfg.get("iou", 0.5),
            device=0,
            plots=False,
            save_json=False,
            project=str(tmp),
            name="eval",
            verbose=False,
        )
        return {
            "mAP50":     round(float(metrics.box.map50), 4),
            "mAP50-95":  round(float(metrics.box.map),   4),
            "Precision": round(float(metrics.box.mp),    4),
            "Recall":    round(float(metrics.box.mr),    4),
        }
    except Exception as e:
        print(f"    [warn] YOLO eval failed: {e}")
        return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Blind test on an external image folder")
    parser.add_argument("--config",      required=True, help="Path to retrain config YAML")
    parser.add_argument("--test-images", required=True, help="Folder of test images")
    parser.add_argument("--labels",      default=None,  help="Folder of YOLO .txt labels (optional)")
    args = parser.parse_args()

    cfg_path = resolve(args.config)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    class_names = cfg["classes"]
    workspace_dir = resolve(cfg["workspace_dir"])
    val_cfg = cfg.get("validate", {})

    test_images_dir = Path(args.test_images).resolve()
    labels_dir = Path(args.labels).resolve() if args.labels else None

    if not test_images_dir.exists():
        raise SystemExit(f"Error: --test-images path does not exist: {test_images_dir}")
    if labels_dir and not labels_dir.exists():
        raise SystemExit(f"Error: --labels path does not exist: {labels_dir}")

    test_imgs = sorted(
        p for p in test_images_dir.iterdir() if p.suffix.lower() in SUPPORTED_IMG
    )
    if not test_imgs:
        raise SystemExit(f"Error: no images found in {test_images_dir}")

    runs_dir = workspace_dir / "runs"
    if not runs_dir.exists():
        raise SystemExit(f"Error: no runs directory found at {runs_dir}")

    run_dirs = sorted(
        d for d in runs_dir.iterdir()
        if d.is_dir() and (
            (d / "weights" / "best.pt").exists() or
            (d / "rfdetr" / "checkpoint_best_total.pth").exists()
        )
    )
    if not run_dirs:
        raise SystemExit(f"Error: no trained models found in {runs_dir}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = workspace_dir / f"external_blind_test_{timestamp}"
    out_root.mkdir(parents=True)
    tmp_root = out_root / ".tmp_dataset"

    print(f"\nBlind test — {len(test_imgs)} images, {len(run_dirs)} run(s)")
    print(f"Output → {out_root}\n")

    rows = []
    for run_dir in run_dirs:
        print(f"  [{run_dir.name}]")
        run_out = out_root / run_dir.name
        has_yolo   = (run_dir / "weights" / "best.pt").exists()
        has_rfdetr = (run_dir / "rfdetr" / "checkpoint_best_total.pth").exists()

        yolo_row = {"run": run_dir.name, "framework": "yolo",
                    "mAP50": "N/A", "mAP50-95": "N/A", "Precision": "N/A", "Recall": "N/A"}
        rfdetr_row = {"run": run_dir.name, "framework": "rfdetr",
                      "mAP50": "N/A", "mAP50-95": "N/A", "Precision": "N/A", "Recall": "N/A"}

        if has_yolo:
            _detect_yolo(run_dir, test_imgs, run_out / "yolo", val_cfg)
            if labels_dir:
                m = _eval_yolo_metrics(run_dir, test_images_dir, labels_dir,
                                       class_names, val_cfg, tmp_root)
                if m:
                    yolo_row.update(m)
            rows.append(yolo_row)

        if has_rfdetr:
            _detect_rfdetr(run_dir, test_imgs, run_out / "rfdetr", val_cfg, class_names)
            rows.append(rfdetr_row)

    shutil.rmtree(tmp_root, ignore_errors=True)

    # Summary table
    headers = ["run", "framework", "mAP50", "mAP50-95", "Precision", "Recall"]
    col_w = [max(len(h), max(len(str(r[h])) for r in rows)) for h in headers]
    sep = "  ".join("-" * w for w in col_w)
    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, col_w))
    print(f"\n{'=' * len(sep)}")
    print(header_line)
    print(sep)
    for r in rows:
        print("  ".join(str(r[h]).ljust(w) for h, w in zip(headers, col_w)))
    print(f"{'=' * len(sep)}\n")

    summary_csv = out_root / "summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved → {summary_csv}")


if __name__ == "__main__":
    main()
