"""
Iterative retraining pipeline — YOLO + RF-DETR.

Workflow:
  1.  Scan datasets/        — find batch folders, verify class consistency
  2.  Merge (YOLO)          — symlink all batches into workspace/merged/
  2b. Convert to COCO       — workspace/merged_coco/ for RF-DETR
  3.  Train YOLO            — new run_NNN_... in workspace/runs/
  3b. Train RF-DETR         — saved under run_dir/rfdetr/
  4.  Evaluate ALL models   — YOLO (.pt) + RF-DETR (.pth) on current val set
  5.  Update leaderboard    — workspace/leaderboard.csv (with framework column)
  6.  Detect                — all models on test image folder
  7.  Summary               — ranked table across both frameworks

Usage:
  .venv/bin/python scripts/retrain.py --config configs/retrain/ppe.yaml
  .venv/bin/python scripts/retrain.py --config configs/retrain/ppe.yaml --eval-only
  .venv/bin/python scripts/retrain.py --config configs/retrain/ppe.yaml --skip-rfdetr
  .venv/bin/python scripts/retrain.py --config configs/retrain/ppe.yaml --epochs 100
"""

import argparse
import csv
import json
import os

# Allow ultralytics MLflow callback to use local file store without migration error
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

from collections import defaultdict
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def banner(msg):
    bar = "=" * 62
    print(f"\n{bar}\n  {msg}\n{bar}")


# ─── Phase 1: Scan batch folders ─────────────────────────────────

def scan_batches(datasets_dir: Path, expected_classes: list) -> list:
    batches = []
    for d in sorted(datasets_dir.iterdir()):
        if not d.is_dir():
            continue
        yaml_path = d / "dataset.yaml"
        if not yaml_path.exists():
            continue
        with open(yaml_path) as f:
            ds = yaml.safe_load(f)
        names = ds.get("names", [])
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names)]
        if names != expected_classes:
            raise ValueError(
                f"Class mismatch in {d.name}:\n"
                f"  Expected : {expected_classes}\n"
                f"  Found    : {names}\n"
                f"All batches must have identical class names in the same order."
            )
        batches.append(d)
    if not batches:
        raise FileNotFoundError(
            f"No valid dataset batches found in {datasets_dir}.\n"
            f"Each batch folder must contain dataset.yaml."
        )
    return batches


# ─── Phase 2: Merge YOLO dataset using symlinks ──────────────────

def merge_dataset(batches: list, workspace_dir: Path, expected_classes: list):
    merged_dir = workspace_dir / "merged"
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            d = merged_dir / split / sub
            d.mkdir(parents=True, exist_ok=True)
            for f in d.iterdir():
                if f.is_symlink():
                    f.unlink()

    train_count = val_count = 0
    for batch_dir in batches:
        prefix = batch_dir.name + "__"
        for split in ("train", "val"):
            img_src = batch_dir / split / "images"
            lbl_src = batch_dir / split / "labels"
            if not img_src.exists():
                continue
            for img_path in sorted(img_src.iterdir()):
                if img_path.suffix.lower() not in SUPPORTED_IMG:
                    continue
                dst_img = merged_dir / split / "images" / (prefix + img_path.name)
                dst_lbl = merged_dir / split / "labels" / (prefix + img_path.stem + ".txt")
                src_lbl = lbl_src / (img_path.stem + ".txt")
                if not dst_img.exists():
                    os.symlink(img_path.resolve(), dst_img)
                if src_lbl.exists() and not dst_lbl.exists():
                    os.symlink(src_lbl.resolve(), dst_lbl)
                if split == "train":
                    train_count += 1
                else:
                    val_count += 1

    merged_yaml = {
        "path": str(merged_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "nc": len(expected_classes),
        "names": expected_classes,
    }
    with open(merged_dir / "merged.yaml", "w") as f:
        yaml.dump(merged_yaml, f, default_flow_style=False)

    return merged_dir, train_count, val_count


# ─── Phase 2b: Convert merged YOLO → COCO (for RF-DETR) ──────────

def phase_convert_coco(merged_dir: Path, workspace_dir: Path, expected_classes: list) -> Path:
    from PIL import Image as PILImage

    coco_dir = workspace_dir / "merged_coco"

    # Skip if already up to date (val image count matches)
    val_img_count = sum(
        1 for p in (merged_dir / "val" / "images").iterdir()
        if p.suffix.lower() in SUPPORTED_IMG
    )
    valid_ann = coco_dir / "valid" / "_annotations.coco.json"
    if valid_ann.exists():
        try:
            with open(valid_ann) as f:
                existing = json.load(f)
            if len(existing.get("images", [])) == val_img_count:
                print(f"  COCO dataset up to date ({val_img_count} val images) — skipping conversion.")
                return coco_dir
        except Exception:
            pass

    categories = [
        {"id": i, "name": name, "supercategory": "object"}
        for i, name in enumerate(expected_classes)
    ]

    for src_split, dst_split in [("train", "train"), ("val", "valid")]:
        img_src = merged_dir / src_split / "images"
        lbl_src = merged_dir / src_split / "labels"
        dst_dir  = coco_dir / dst_split
        dst_dir.mkdir(parents=True, exist_ok=True)

        # Clean existing symlinks
        for f in dst_dir.iterdir():
            if f.is_symlink():
                f.unlink()

        coco: dict = {"images": [], "annotations": [], "categories": categories}
        ann_id = 1

        for img_id, img_path in enumerate(sorted(img_src.iterdir()), 1):
            if img_path.suffix.lower() not in SUPPORTED_IMG:
                continue

            with PILImage.open(img_path) as pil_img:
                W, H = pil_img.size

            coco["images"].append({
                "id": img_id, "file_name": img_path.name, "width": W, "height": H,
            })

            dst_link = dst_dir / img_path.name
            if not dst_link.exists():
                os.symlink(img_path.resolve(), dst_link)

            lbl_path = lbl_src / (img_path.stem + ".txt")
            if lbl_path.exists():
                with open(lbl_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        cls_id = int(parts[0])
                        cx, cy, bw, bh = (float(parts[1]), float(parts[2]),
                                          float(parts[3]), float(parts[4]))
                        x_min = (cx - bw / 2) * W
                        y_min = (cy - bh / 2) * H
                        w_abs = bw * W
                        h_abs = bh * H
                        coco["annotations"].append({
                            "id":          ann_id,
                            "image_id":    img_id,
                            "category_id": cls_id,
                            "bbox":        [round(x_min, 2), round(y_min, 2),
                                            round(w_abs, 2), round(h_abs, 2)],
                            "area":        round(w_abs * h_abs, 2),
                            "iscrowd":     0,
                        })
                        ann_id += 1

        ann_path = dst_dir / "_annotations.coco.json"
        with open(ann_path, "w") as f:
            json.dump(coco, f)

        print(f"  {dst_split}: {len(coco['images'])} images, {len(coco['annotations'])} annotations")

    return coco_dir


# ─── Phase 3: Train YOLO ─────────────────────────────────────────

def _next_run_number(runs_dir: Path) -> int:
    nums = []
    if runs_dir.exists():
        for d in runs_dir.iterdir():
            parts = d.name.split("_")
            if d.is_dir() and d.name.startswith("run_") and len(parts) >= 2:
                try:
                    nums.append(int(parts[1]))
                except ValueError:
                    pass
    return (max(nums) + 1) if nums else 1


def phase_train_yolo(train_cfg: dict, merged_dir: Path, batches: list,
                     train_count: int, val_count: int, runs_dir: Path) -> tuple:
    from ultralytics import YOLO

    run_num   = _next_run_number(runs_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = f"run_{run_num:03d}_{timestamp}"

    banner(f"Phase 3 — Training YOLO: {run_name}")

    model = YOLO(str(resolve(train_cfg["base_model"])))
    model.train(
        data=str(merged_dir / "merged.yaml"),
        epochs=train_cfg.get("epochs", 100),
        imgsz=train_cfg.get("imgsz", 640),
        batch=train_cfg.get("batch", 16),
        device=0,
        freeze=train_cfg.get("freeze", 0),
        lr0=train_cfg.get("lr0", 0.01),
        patience=train_cfg.get("patience", 50),
        project=str(runs_dir),
        name=run_name,
    )

    run_dir = runs_dir / run_name
    meta = {
        "run":          run_name,
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "batches":      [b.name for b in batches],
        "train_images": train_count,
        "val_images":   val_count,
    }
    with open(run_dir / "run_meta.yaml", "w") as f:
        yaml.dump(meta, f)

    return run_name, run_dir


# ─── Phase 3b: Train RF-DETR ─────────────────────────────────────

_RFDETR_CLASS_MAP = {
    "base":   "RFDETRBase",
    "large":  "RFDETRLarge",
    "small":  "RFDETRSmall",
    "medium": "RFDETRMedium",
    "nano":   "RFDETRNano",
}


def phase_train_rfdetr(rfdetr_cfg: dict, merged_coco_dir: Path, run_dir: Path):
    import importlib
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="rfdetr")

    banner(f"Phase 3b — Training RF-DETR: {run_dir.name}")

    model_type = rfdetr_cfg.get("model", "base").lower()
    cls_name   = _RFDETR_CLASS_MAP.get(model_type, "RFDETRBase")
    ModelClass = getattr(importlib.import_module("rfdetr"), cls_name)
    model      = ModelClass()
    model.train(
        dataset_dir=str(merged_coco_dir),
        epochs=rfdetr_cfg.get("epochs", 50),
        batch_size=rfdetr_cfg.get("batch_size", 4),
        grad_accum_steps=rfdetr_cfg.get("grad_accum_steps", 4),
        lr=rfdetr_cfg.get("lr", 1e-4),
        resolution=rfdetr_cfg.get("resolution", 560),
        output_dir=str(run_dir / "rfdetr"),
    )


# ─── Phase 4: Evaluate all models (YOLO + RF-DETR) ───────────────

def _load_meta(run_dir: Path) -> dict:
    meta_path = run_dir / "run_meta.yaml"
    if meta_path.exists():
        with open(meta_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def _eval_yolo(run_dir: Path, merged_dir: Path, val_cfg: dict) -> dict | None:
    from ultralytics import YOLO

    weights = run_dir / "weights" / "best.pt"
    if not weights.exists():
        return None

    meta    = _load_meta(run_dir)
    metrics = YOLO(str(weights)).val(
        data=str(merged_dir / "merged.yaml"),
        split="val",
        imgsz=val_cfg.get("imgsz", 640),
        batch=val_cfg.get("batch", 16),
        device=0,
        conf=val_cfg.get("conf", 0.5),
        iou=val_cfg.get("iou", 0.5),
        plots=False,
        save_json=False,
        project=str(run_dir),
        name="eval_yolo",
        verbose=False,
    )
    return {
        "run":          run_dir.name,
        "framework":    "yolo",
        "timestamp":    meta.get("timestamp", ""),
        "batches":      ",".join(meta.get("batches", [])),
        "train_images": meta.get("train_images", ""),
        "val_images":   meta.get("val_images", ""),
        "mAP50":        round(float(metrics.box.map50), 4),
        "mAP50-95":     round(float(metrics.box.map),   4),
        "Precision":    round(float(metrics.box.mp),    4),
        "Recall":       round(float(metrics.box.mr),    4),
        "weights":      str(weights),
        "is_best":      False,
    }


def _eval_rfdetr(run_dir: Path, merged_coco_dir: Path) -> dict | None:
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="rfdetr")
    from rfdetr import RFDETR
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    weights = run_dir / "rfdetr" / "checkpoint_best_total.pth"
    if not weights.exists():
        return None

    val_dir  = merged_coco_dir / "valid"
    ann_file = val_dir / "_annotations.coco.json"
    if not ann_file.exists():
        return None

    meta    = _load_meta(run_dir)
    model   = RFDETR.from_checkpoint(str(weights))
    coco_gt = COCO(str(ann_file))
    preds   = []

    for img_id in coco_gt.getImgIds():
        info     = coco_gt.loadImgs(img_id)[0]
        img_path = val_dir / info["file_name"]
        dets     = model.predict(str(img_path), threshold=0.001)
        if dets is None or len(dets) == 0:
            continue
        for i in range(len(dets.xyxy)):
            x1, y1, x2, y2 = dets.xyxy[i]
            preds.append({
                "image_id":    img_id,
                "category_id": int(dets.class_id[i]),
                "bbox":        [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score":       float(dets.confidence[i]),
            })

    if not preds:
        map50 = map5095 = 0.0
    else:
        coco_dt   = coco_gt.loadRes(preds)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        map5095, map50 = float(coco_eval.stats[0]), float(coco_eval.stats[1])

    return {
        "run":          run_dir.name,
        "framework":    "rfdetr",
        "timestamp":    meta.get("timestamp", ""),
        "batches":      ",".join(meta.get("batches", [])),
        "train_images": meta.get("train_images", ""),
        "val_images":   meta.get("val_images", ""),
        "mAP50":        round(map50,   4),
        "mAP50-95":     round(map5095, 4),
        "Precision":    "N/A",
        "Recall":       "N/A",
        "weights":      str(weights),
        "is_best":      False,
    }


def phase_evaluate_all(runs_dir: Path, merged_dir: Path,
                       merged_coco_dir: Path | None, val_cfg: dict) -> list:
    banner("Phase 4 — Evaluating all models (YOLO + RF-DETR)")
    run_dirs = sorted(d for d in runs_dir.iterdir() if d.is_dir())
    results  = []
    total    = len(run_dirs)

    for i, run_dir in enumerate(run_dirs, 1):
        has_yolo   = (run_dir / "weights" / "best.pt").exists()
        has_rfdetr = (run_dir / "rfdetr"  / "checkpoint_best_total.pth").exists()

        if has_yolo:
            print(f"  [{i}/{total}] {run_dir.name} [yolo] ...")
            r = _eval_yolo(run_dir, merged_dir, val_cfg)
            if r:
                results.append(r)

        if has_rfdetr and merged_coco_dir:
            print(f"  [{i}/{total}] {run_dir.name} [rfdetr] ...")
            r = _eval_rfdetr(run_dir, merged_coco_dir)
            if r:
                results.append(r)

    return results


# ─── Phase 5: Update leaderboard ─────────────────────────────────

def update_leaderboard(workspace_dir: Path, eval_results: list) -> list:
    lb_path = workspace_dir / "leaderboard.csv"
    headers = [
        "run", "framework", "timestamp", "batches", "train_images", "val_images",
        "mAP50", "mAP50-95", "Precision", "Recall", "weights", "is_best",
    ]
    if eval_results:
        best = max(
            (r for r in eval_results if isinstance(r.get("mAP50"), float)),
            key=lambda r: r["mAP50"],
            default=None,
        )
        for r in eval_results:
            r["is_best"] = bool(
                best and r["run"] == best["run"] and r["framework"] == best["framework"]
            )
    _save_csv(eval_results, lb_path, headers)
    return eval_results


# ─── Phase 6: Detect on image folder (YOLO + RF-DETR) ────────────

def _detect_yolo(run_dir: Path, images: list, detect_cfg: dict) -> dict | None:
    from ultralytics import YOLO

    weights = run_dir / "weights" / "best.pt"
    if not weights.exists():
        return None

    conf    = detect_cfg.get("conf", 0.5)
    imgsz   = detect_cfg.get("imgsz", 640)
    out_dir = run_dir / "detect_yolo"
    out_dir.mkdir(parents=True, exist_ok=True)

    model       = YOLO(str(weights))
    class_stats = defaultdict(lambda: {"count": 0, "conf_sum": 0.0, "img_names": set()})
    total_det   = 0

    for img_path in images:
        result = model(str(img_path), conf=conf, imgsz=imgsz, verbose=False)[0]
        for box in result.boxes:
            cls_name = model.names[int(box.cls)]
            cs = float(box.conf)
            class_stats[cls_name]["count"]   += 1
            class_stats[cls_name]["conf_sum"] += cs
            class_stats[cls_name]["img_names"].add(img_path.name)
            total_det += 1
        result.save(str(out_dir / img_path.name))

    return _build_det_result(run_dir, "yolo", images, class_stats, total_det, out_dir)


def _detect_rfdetr(run_dir: Path, images: list, detect_cfg: dict,
                   class_names: list) -> dict | None:
    import warnings
    import cv2
    import numpy as np
    import supervision as sv
    warnings.filterwarnings("ignore", category=FutureWarning, module="rfdetr")
    from rfdetr import RFDETR

    weights = run_dir / "rfdetr" / "checkpoint_best_total.pth"
    if not weights.exists():
        return None

    conf    = detect_cfg.get("conf", 0.5)
    out_dir = run_dir / "detect_rfdetr"
    out_dir.mkdir(parents=True, exist_ok=True)

    model           = RFDETR.from_checkpoint(str(weights))
    box_annotator   = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    class_stats     = defaultdict(lambda: {"count": 0, "conf_sum": 0.0, "img_names": set()})
    total_det       = 0

    for img_path in images:
        dets = model.predict(str(img_path), threshold=conf)

        if dets is not None and len(dets) > 0:
            for i in range(len(dets.xyxy)):
                cls_id   = int(dets.class_id[i])
                cls_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
                cs       = float(dets.confidence[i])
                class_stats[cls_name]["count"]   += 1
                class_stats[cls_name]["conf_sum"] += cs
                class_stats[cls_name]["img_names"].add(img_path.name)
                total_det += 1

        img = cv2.imread(str(img_path))
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

    return _build_det_result(run_dir, "rfdetr", images, class_stats, total_det, out_dir)


def _build_det_result(run_dir: Path, framework: str, images: list,
                      class_stats: dict, total_det: int, out_dir: Path) -> dict:
    meta             = _load_meta(run_dir)
    images_with_det  = len({n for s in class_stats.values() for n in s["img_names"]})
    top_class        = max(class_stats, key=lambda c: class_stats[c]["count"]) if class_stats else "-"
    top_label        = f"{top_class}({class_stats[top_class]['count']})" if class_stats else "-"
    return {
        "run":             run_dir.name,
        "framework":       framework,
        "timestamp":       meta.get("timestamp", ""),
        "total_det":       total_det,
        "images_with_det": images_with_det,
        "num_images":      len(images),
        "top_class":       top_label,
        "annotated_dir":   str(out_dir),
    }


def phase_detect_all(runs_dir: Path, detect_cfg: dict,
                     workspace_dir: Path, class_names: list) -> list:
    banner("Phase 6 — Detection on image folder")
    images_path = resolve(detect_cfg["images"])
    images      = sorted(p for p in images_path.iterdir() if p.suffix.lower() in SUPPORTED_IMG)
    if not images:
        print(f"  No images found in {detect_cfg['images']} — skipping.")
        return []
    print(f"  Found {len(images)} images in {detect_cfg['images']}")

    run_dirs    = sorted(d for d in runs_dir.iterdir() if d.is_dir())
    det_results = []

    for i, run_dir in enumerate(run_dirs, 1):
        has_yolo   = (run_dir / "weights" / "best.pt").exists()
        has_rfdetr = (run_dir / "rfdetr"  / "checkpoint_best_total.pth").exists()

        if has_yolo:
            print(f"  [{i}/{len(run_dirs)}] {run_dir.name} [yolo] ...")
            r = _detect_yolo(run_dir, images, detect_cfg)
            if r:
                det_results.append(r)

        if has_rfdetr:
            print(f"  [{i}/{len(run_dirs)}] {run_dir.name} [rfdetr] ...")
            r = _detect_rfdetr(run_dir, images, detect_cfg, class_names)
            if r:
                det_results.append(r)

    hist_path = workspace_dir / "detection_history.csv"
    _save_csv(
        det_results,
        hist_path,
        ["run", "framework", "timestamp", "total_det",
         "images_with_det", "num_images", "top_class", "annotated_dir"],
    )
    print(f"\n  Saved detection history → {hist_path}")
    return det_results


# ─── Phase 7: Print ranked summary ───────────────────────────────

def _fmt_metric(val, best_val):
    if not isinstance(val, float):
        return f"{'N/A':>8}"
    star = "*" if val == best_val else " "
    return f"{val:>7.4f}{star}"


def phase_summary(
    eval_results: list,
    det_results:  list,
    current_run:  str | None,
    batches:      list,
    train_count:  int,
    val_count:    int,
    detect_cfg:   dict,
    workspace_dir: Path,
):
    if not eval_results:
        print("\nNo evaluation results to summarize.")
        return

    sorted_val = sorted(
        eval_results,
        key=lambda r: r["mAP50"] if isinstance(r.get("mAP50"), float) else 0,
        reverse=True,
    )
    best_row   = sorted_val[0]
    MCOLS      = ["mAP50", "mAP50-95", "Precision", "Recall"]
    best_vals  = {
        m: max((r[m] for r in eval_results if isinstance(r.get(m), float)), default=None)
        for m in MCOLS
    }

    batch_label = "+".join(b.name for b in batches) if batches else "—"
    banner(f"RETRAINING SUMMARY — {current_run or 'eval-only'}")
    print(f"  Batches  : {batch_label}")
    if batches:
        print(f"  Dataset  : {train_count} train / {val_count} val images")
    if "images" in detect_cfg and det_results:
        print(f"  Test imgs: {detect_cfg['images']} ({det_results[0]['num_images']} images)")
    print(f"  Leaderboard → {workspace_dir / 'leaderboard.csv'}")

    RUN_W = 34
    FRM_W = 6

    print(f"\n--- VALIDATION (labeled val set) ---")
    hdr = (f"{'Rank':<5}  {'Run':{RUN_W}}  {'Frm':{FRM_W}}  {'Batches':<18}  "
           f"{'mAP50':>8}  {'mAP50-95':>9}  {'Prec':>8}  {'Recall':>8}")
    print(hdr)
    print("-" * len(hdr))

    for rank, r in enumerate(sorted_val, 1):
        is_cur  = (r["run"] == current_run)
        is_best = (r["run"] == best_row["run"] and r["framework"] == best_row["framework"])
        star    = "★" if is_best else " "
        suffix  = " [NEW]" if is_cur else ""
        run_str = f"{star} {r['run']}{suffix}"[:RUN_W]
        frm_str = r.get("framework", "")[:FRM_W]
        bat_str = r.get("batches", "")[:16]
        print(
            f"{rank:<5}  {run_str:{RUN_W}}  {frm_str:{FRM_W}}  {bat_str:<18}  "
            f"{_fmt_metric(r.get('mAP50'),    best_vals['mAP50'])}  "
            f"{_fmt_metric(r.get('mAP50-95'), best_vals['mAP50-95']):>9}  "
            f"{_fmt_metric(r.get('Precision'),best_vals['Precision']):>8}  "
            f"{_fmt_metric(r.get('Recall'),   best_vals['Recall']):>8}"
        )

    print(f"\n  ★ = overall best   * = best value in column   [NEW] = this run")

    if det_results:
        sorted_det   = sorted(det_results, key=lambda r: r["total_det"], reverse=True)
        best_det_run = sorted_det[0]
        print(f"\n--- DETECTION (image folder: {detect_cfg.get('images', '—')}) ---")
        hdr2 = (f"{'Rank':<5}  {'Run':{RUN_W}}  {'Frm':{FRM_W}}  "
                f"{'Total Det':>10}  {'Imgs w/ Det':>12}  {'Top Class'}")
        print(hdr2)
        print("-" * len(hdr2))
        for rank, r in enumerate(sorted_det, 1):
            is_best_d = (r["run"] == best_det_run["run"]
                         and r["framework"] == best_det_run["framework"])
            star2  = "★" if is_best_d else " "
            suffix = " [NEW]" if r["run"] == current_run else ""
            run_str = f"{star2} {r['run']}{suffix}"[:RUN_W]
            frm_str = r.get("framework", "")[:FRM_W]
            d_star  = "*" if is_best_d else " "
            img_frac = f"{r['images_with_det']}/{r['num_images']}"
            print(f"{rank:<5}  {run_str:{RUN_W}}  {frm_str:{FRM_W}}  "
                  f"{r['total_det']:>9}{d_star}  {img_frac:>12}  {r['top_class']}")
        print(f"\n  ★ = most detections   * = best total")

    print(f"\n{'='*62}")
    print(f"  Best model  : {best_row['framework'].upper()}  mAP50={best_row['mAP50']}")
    print(f"  Weights     : {best_row['weights']}")
    print(f"{'='*62}\n")


# ─── CSV helper ──────────────────────────────────────────────────

def _save_csv(rows: list, path: Path, headers: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ─── Entry point ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Iterative retraining: merge → train YOLO + RF-DETR → evaluate all → leaderboard"
    )
    parser.add_argument("--config",      required=True, help="Path to retrain config YAML")
    parser.add_argument("--eval-only",   action="store_true",
                        help="Skip training; re-evaluate all existing models")
    parser.add_argument("--skip-rfdetr", action="store_true",
                        help="Skip RF-DETR training this run")
    parser.add_argument("--epochs",      type=int, help="Override YOLO epochs from config")
    args = parser.parse_args()

    with open(resolve(args.config)) as f:
        cfg = yaml.safe_load(f)

    if args.epochs:
        cfg.setdefault("train", {})["epochs"] = args.epochs

    datasets_dir  = resolve(cfg.get("datasets_dir", "datasets"))
    workspace_dir = resolve(cfg.get("workspace_dir", "workspace"))
    runs_dir      = workspace_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    expected_classes = cfg["classes"]
    train_cfg   = cfg.get("train",        {})
    rfdetr_cfg  = cfg.get("rfdetr_train", {})
    val_cfg     = cfg.get("validate",     {})
    detect_cfg  = cfg.get("detect",       {})

    run_rfdetr = (
        not args.eval_only
        and not args.skip_rfdetr
        and rfdetr_cfg.get("enabled", True)
    )

    # Phase 1 — Scan batches
    banner("Phase 1 — Scanning dataset batches")
    batches = scan_batches(datasets_dir, expected_classes)
    print(f"  Found {len(batches)} batch(es): {[b.name for b in batches]}")

    # Phase 2 — Merge YOLO dataset
    banner("Phase 2 — Merging dataset (symlinks)")
    merged_dir, train_count, val_count = merge_dataset(batches, workspace_dir, expected_classes)
    print(f"  {train_count} train images, {val_count} val images → {merged_dir}")

    # Phase 2b — Convert to COCO (for RF-DETR)
    merged_coco_dir = None
    if run_rfdetr or any(
        (d / "rfdetr" / "checkpoint_best_total.pth").exists()
        for d in runs_dir.iterdir() if d.is_dir()
    ):
        banner("Phase 2b — Converting merged dataset to COCO (RF-DETR)")
        merged_coco_dir = phase_convert_coco(merged_dir, workspace_dir, expected_classes)
        print(f"  COCO dataset → {merged_coco_dir}")

    # Phase 3 — Train YOLO
    current_run = None
    run_dir     = None
    if not args.eval_only:
        current_run, run_dir = phase_train_yolo(
            train_cfg, merged_dir, batches, train_count, val_count, runs_dir
        )
    else:
        banner("Phase 3 — Skipped (--eval-only)")

    # Phase 3b — Train RF-DETR
    if run_rfdetr and run_dir is not None:
        phase_train_rfdetr(rfdetr_cfg, merged_coco_dir, run_dir)
    elif args.eval_only:
        banner("Phase 3b — Skipped (--eval-only)")
    elif args.skip_rfdetr or not rfdetr_cfg.get("enabled", True):
        banner("Phase 3b — Skipped (--skip-rfdetr / disabled in config)")

    # Phase 4 — Evaluate all models
    eval_results = phase_evaluate_all(runs_dir, merged_dir, merged_coco_dir, val_cfg)
    if not eval_results:
        print("\n  No trained models found. Run without --eval-only first.")
        return

    # Phase 5 — Leaderboard
    banner("Phase 5 — Updating leaderboard")
    eval_results = update_leaderboard(workspace_dir, eval_results)
    print(f"  Saved → {workspace_dir / 'leaderboard.csv'}")

    # Phase 6 — Detect
    det_results: list = []
    if "images" in detect_cfg:
        det_results = phase_detect_all(runs_dir, detect_cfg, workspace_dir, expected_classes)
    else:
        banner("Phase 6 — Skipped (no detect.images in config)")

    # Phase 7 — Summary
    phase_summary(
        eval_results, det_results, current_run,
        batches, train_count, val_count, detect_cfg, workspace_dir,
    )


if __name__ == "__main__":
    main()
