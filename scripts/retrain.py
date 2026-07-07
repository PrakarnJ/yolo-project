"""
Iterative retraining pipeline — YOLO + RF-DETR.

Workflow:
  1.  Scan datasets/        — find batch folders, verify class consistency
  2.  Merge (YOLO)          — symlink all batches into workspace/merged/
                             (val images split: 90% val / 10% test via manifest)
  2b. Convert to COCO       — workspace/merged_coco/ for RF-DETR (train + valid + test)
  3.  Train YOLO            — new run_NNN_... in workspace/runs/
  3b. Train RF-DETR         — saved under run_dir/rfdetr/
  4.  Evaluate ALL models   — YOLO (.pt) + RF-DETR (.pth) on current val set
  5.  Update leaderboard    — workspace/leaderboard.csv (with framework column)
  6.  Blind test            — all models on held-out test split; cleanup test dirs
  7.  Summary               — val + blind test ranked tables across both frameworks

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
import shutil

# Allow ultralytics MLflow callback to use local file store without migration error
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

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

def _load_manifest(workspace_dir: Path) -> dict:
    manifest_path = workspace_dir / "test_split_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)
    return {}


def _save_manifest(workspace_dir: Path, manifest: dict):
    workspace_dir.mkdir(parents=True, exist_ok=True)
    with open(workspace_dir / "test_split_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def merge_dataset(batches: list, workspace_dir: Path, expected_classes: list):
    merged_dir = workspace_dir / "merged"
    for split in ("train", "val", "test"):
        for sub in ("images", "labels"):
            d = merged_dir / split / sub
            d.mkdir(parents=True, exist_ok=True)
            for f in d.iterdir():
                if f.is_symlink():
                    f.unlink()

    manifest = _load_manifest(workspace_dir)
    manifest_updated = False
    train_count = val_count = test_count = 0

    for batch_dir in batches:
        prefix = batch_dir.name + "__"

        # ── Train split ───────────────────────────────────────────
        img_src = batch_dir / "train" / "images"
        lbl_src = batch_dir / "train" / "labels"
        if img_src.exists():
            for img_path in sorted(img_src.iterdir()):
                if img_path.suffix.lower() not in SUPPORTED_IMG:
                    continue
                dst_img = merged_dir / "train" / "images" / (prefix + img_path.name)
                dst_lbl = merged_dir / "train" / "labels" / (prefix + img_path.stem + ".txt")
                src_lbl = lbl_src / (img_path.stem + ".txt")
                if not dst_img.exists():
                    os.symlink(img_path.resolve(), dst_img)
                if src_lbl.exists() and not dst_lbl.exists():
                    os.symlink(src_lbl.resolve(), dst_lbl)
                train_count += 1

        # ── Val/test split — carve 10% of val into test via manifest ─
        val_img_src = batch_dir / "val" / "images"
        val_lbl_src = batch_dir / "val" / "labels"
        if not val_img_src.exists():
            continue

        all_val_imgs = sorted(
            p for p in val_img_src.iterdir() if p.suffix.lower() in SUPPORTED_IMG
        )

        if batch_dir.name not in manifest:
            # First time: take every 10th image as test (deterministic, no seed needed)
            test_names = set(p.name for p in all_val_imgs[::10])
            manifest[batch_dir.name] = sorted(test_names)
            manifest_updated = True
        else:
            test_names = set(manifest[batch_dir.name])

        for img_path in all_val_imgs:
            dest_split = "test" if img_path.name in test_names else "val"
            dst_img = merged_dir / dest_split / "images" / (prefix + img_path.name)
            dst_lbl = merged_dir / dest_split / "labels" / (prefix + img_path.stem + ".txt")
            src_lbl = val_lbl_src / (img_path.stem + ".txt")
            if not dst_img.exists():
                os.symlink(img_path.resolve(), dst_img)
            if src_lbl.exists() and not dst_lbl.exists():
                os.symlink(src_lbl.resolve(), dst_lbl)
            if dest_split == "test":
                test_count += 1
            else:
                val_count += 1

    if manifest_updated:
        _save_manifest(workspace_dir, manifest)

    merged_yaml = {
        "path":  str(merged_dir.resolve()),
        "train": "train/images",
        "val":   "val/images",
        "test":  "test/images",
        "nc":    len(expected_classes),
        "names": expected_classes,
    }
    with open(merged_dir / "merged.yaml", "w") as f:
        yaml.dump(merged_yaml, f, default_flow_style=False)

    return merged_dir, train_count, val_count, test_count


# ─── Phase 2b: Convert merged YOLO → COCO (for RF-DETR) ──────────

def phase_convert_coco(merged_dir: Path, workspace_dir: Path, expected_classes: list) -> Path:
    from PIL import Image as PILImage

    coco_dir = workspace_dir / "merged_coco"

    # Skip if already up to date (check val and test image counts against existing JSONs)
    val_img_count = sum(
        1 for p in (merged_dir / "val" / "images").iterdir()
        if p.suffix.lower() in SUPPORTED_IMG
    )
    test_img_dir = merged_dir / "test" / "images"
    test_img_count = sum(
        1 for p in test_img_dir.iterdir() if p.suffix.lower() in SUPPORTED_IMG
    ) if test_img_dir.exists() else 0

    valid_ann = coco_dir / "valid" / "_annotations.coco.json"
    test_ann  = coco_dir / "test"  / "_annotations.coco.json"
    if valid_ann.exists() and test_ann.exists():
        try:
            with open(valid_ann) as f:
                existing_val = json.load(f)
            with open(test_ann) as f:
                existing_test = json.load(f)
            if (len(existing_val.get("images", [])) == val_img_count and
                    len(existing_test.get("images", [])) == test_img_count):
                print(f"  COCO dataset up to date "
                      f"({val_img_count} val, {test_img_count} test images) — skipping conversion.")
                return coco_dir
        except Exception:
            pass

    categories = [
        {"id": i, "name": name, "supercategory": "object"}
        for i, name in enumerate(expected_classes)
    ]

    for src_split, dst_split in [("train", "train"), ("val", "valid"), ("test", "test")]:
        src_img_dir = merged_dir / src_split / "images"
        if not src_img_dir.exists():
            continue
        src_lbl_dir = merged_dir / src_split / "labels"
        dst_dir = coco_dir / dst_split
        dst_dir.mkdir(parents=True, exist_ok=True)

        # Clean existing symlinks
        for f in dst_dir.iterdir():
            if f.is_symlink():
                f.unlink()

        coco: dict = {"images": [], "annotations": [], "categories": categories}
        ann_id = 1

        for img_id, img_path in enumerate(sorted(src_img_dir.iterdir()), 1):
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

            lbl_path = src_lbl_dir / (img_path.stem + ".txt")
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
                     train_count: int, val_count: int, test_count: int,
                     runs_dir: Path) -> tuple:
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
        optimizer=train_cfg.get("optimizer", "auto"),
        seed=train_cfg.get("seed", 0),
        lr0=train_cfg.get("lr0", 0.01),
        patience=train_cfg.get("patience", 50),
        copy_paste=train_cfg.get("copy_paste", 0.0),
        mixup=train_cfg.get("mixup", 0.0),
        erasing=train_cfg.get("erasing", 0.4),
        hsv_v=train_cfg.get("hsv_v", 0.4),
        degrees=train_cfg.get("degrees", 0.0),
        perspective=train_cfg.get("perspective", 0.0),
        scale=train_cfg.get("scale", 0.5),
        translate=train_cfg.get("translate", 0.1),
        shear=train_cfg.get("shear", 0.0),
        mosaic=train_cfg.get("mosaic", 1.0),
        close_mosaic=train_cfg.get("close_mosaic", 10),
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
        "test_images":  test_count,
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


def _eval_yolo(run_dir: Path, merged_dir: Path, val_cfg: dict,
               split: str = "val") -> dict | None:
    from ultralytics import YOLO

    weights = run_dir / "weights" / "best.pt"
    if not weights.exists():
        return None

    meta      = _load_meta(run_dir)
    eval_name = "eval_yolo" if split == "val" else f"eval_yolo_{split}"
    metrics   = YOLO(str(weights)).val(
        data=str(merged_dir / "merged.yaml"),
        split=split,
        imgsz=val_cfg.get("imgsz", 640),
        batch=val_cfg.get("batch", 16),
        device=0,
        conf=val_cfg.get("conf", 0.5),
        iou=val_cfg.get("iou", 0.5),
        plots=False,
        save_json=False,
        project=str(run_dir),
        name=eval_name,
        verbose=False,
    )
    return {
        "run":          run_dir.name,
        "framework":    "yolo",
        "timestamp":    meta.get("timestamp", ""),
        "batches":      ",".join(meta.get("batches", [])),
        "train_images": meta.get("train_images", ""),
        "val_images":   meta.get("val_images", ""),
        "test_images":  meta.get("test_images", ""),
        "mAP50":        round(float(metrics.box.map50), 4),
        "mAP50-95":     round(float(metrics.box.map),   4),
        "Precision":    round(float(metrics.box.mp),    4),
        "Recall":       round(float(metrics.box.mr),    4),
        "weights":      str(weights),
        "is_best":      False,
    }


def _eval_rfdetr(run_dir: Path, merged_coco_dir: Path,
                 coco_split: str = "valid") -> dict | None:
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="rfdetr")
    from rfdetr import RFDETR
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    weights = run_dir / "rfdetr" / "checkpoint_best_total.pth"
    if not weights.exists():
        return None

    val_dir  = merged_coco_dir / coco_split
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
        "test_images":  meta.get("test_images", ""),
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

def _mark_best(results: list) -> list:
    if results:
        best = max(
            (r for r in results if isinstance(r.get("mAP50"), float)),
            key=lambda r: r["mAP50"],
            default=None,
        )
        for r in results:
            r["is_best"] = bool(
                best and r["run"] == best["run"] and r["framework"] == best["framework"]
            )
    return results


def update_leaderboard(workspace_dir: Path, eval_results: list) -> list:
    headers = [
        "run", "framework", "timestamp", "batches", "train_images", "val_images",
        "mAP50", "mAP50-95", "Precision", "Recall", "weights", "is_best",
    ]
    eval_results = _mark_best(eval_results)
    _save_csv(eval_results, workspace_dir / "leaderboard.csv", headers)
    return eval_results


def update_blind_test_leaderboard(workspace_dir: Path, blind_test_results: list) -> list:
    headers = [
        "run", "framework", "timestamp", "batches", "train_images", "val_images", "test_images",
        "mAP50", "mAP50-95", "Precision", "Recall", "weights", "is_best",
    ]
    blind_test_results = _mark_best(blind_test_results)
    _save_csv(blind_test_results, workspace_dir / "blind_test_leaderboard.csv", headers)
    return blind_test_results


# ─── Phase 6: Blind test on held-out test split ───────────────────

def _detect_test_yolo(run_dir: Path, test_imgs: list, val_cfg: dict):
    from ultralytics import YOLO
    weights = run_dir / "weights" / "best.pt"
    if not weights.exists():
        return
    out_dir = run_dir / "detect_test_yolo"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    model = YOLO(str(weights))
    conf  = val_cfg.get("conf", 0.5)
    imgsz = val_cfg.get("imgsz", 640)
    for img_path in test_imgs:
        result = model(str(img_path), conf=conf, imgsz=imgsz, verbose=False)[0]
        result.save(str(out_dir / img_path.name))
    print(f"    → detect_test_yolo/ ({len(test_imgs)} images)")


def _detect_test_rfdetr(run_dir: Path, test_imgs: list, val_cfg: dict, class_names: list):
    import warnings
    import cv2
    import numpy as np
    import supervision as sv
    warnings.filterwarnings("ignore", category=FutureWarning, module="rfdetr")
    from rfdetr import RFDETR

    weights = run_dir / "rfdetr" / "checkpoint_best_total.pth"
    if not weights.exists():
        return
    out_dir = run_dir / "detect_test_rfdetr"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

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
    print(f"    → detect_test_rfdetr/ ({len(test_imgs)} images)")


def phase_blind_test_all(runs_dir: Path, merged_dir: Path,
                          merged_coco_dir: Path | None, val_cfg: dict,
                          workspace_dir: Path, class_names: list) -> list:
    banner("Phase 6 — Blind Test (held-out 10% of val)")

    test_imgs_dir = merged_dir / "test" / "images"
    if not test_imgs_dir.exists() or not any(
        p.suffix.lower() in SUPPORTED_IMG for p in test_imgs_dir.iterdir()
    ):
        print("  No blind test images found — skipping.")
        return []

    test_imgs = sorted(
        p for p in test_imgs_dir.iterdir() if p.suffix.lower() in SUPPORTED_IMG
    )

    run_dirs = sorted(d for d in runs_dir.iterdir() if d.is_dir())
    results  = []
    total    = len(run_dirs)

    for i, run_dir in enumerate(run_dirs, 1):
        has_yolo   = (run_dir / "weights" / "best.pt").exists()
        has_rfdetr = (run_dir / "rfdetr"  / "checkpoint_best_total.pth").exists()

        if has_yolo:
            print(f"  [{i}/{total}] {run_dir.name} [yolo] ...")
            r = _eval_yolo(run_dir, merged_dir, val_cfg, split="test")
            if r:
                results.append(r)

        if has_rfdetr and merged_coco_dir:
            print(f"  [{i}/{total}] {run_dir.name} [rfdetr] ...")
            r = _eval_rfdetr(run_dir, merged_coco_dir, coco_split="test")
            if r:
                results.append(r)

    # Save leaderboard before detection (results are durable even if detection fails)
    results = update_blind_test_leaderboard(workspace_dir, results)
    print(f"  Saved → {workspace_dir / 'blind_test_leaderboard.csv'}")

    # Detection overlays on test images for every run
    print(f"\n  Generating detection overlays on {len(test_imgs)} test images ...")
    for i, run_dir in enumerate(run_dirs, 1):
        print(f"  [{i}/{total}] {run_dir.name}")
        if (run_dir / "weights" / "best.pt").exists():
            _detect_test_yolo(run_dir, test_imgs, val_cfg)
        if (run_dir / "rfdetr" / "checkpoint_best_total.pth").exists():
            _detect_test_rfdetr(run_dir, test_imgs, val_cfg, class_names)

    # One-time cleanup of old unlabeled detect folder (replaced by blind test)
    data_images = ROOT / "data" / "images"
    if data_images.exists():
        shutil.rmtree(data_images)
        print(f"  Cleaned up {data_images}")

    return results


# ─── Phase 6b: Custom blind test (user-supplied folders) ─────────

def ensure_custom_blind_test_dir(datasets_dir: Path) -> Path:
    custom_dir = datasets_dir / "custom_blind_test"
    custom_dir.mkdir(parents=True, exist_ok=True)
    if not discover_custom_test_sets(custom_dir):
        (custom_dir / "default" / "images").mkdir(parents=True, exist_ok=True)
        (custom_dir / "default" / "labels").mkdir(parents=True, exist_ok=True)
    return custom_dir


def discover_custom_test_sets(custom_dir: Path) -> list:
    if not custom_dir.exists():
        return []
    return sorted(
        d for d in custom_dir.iterdir()
        if d.is_dir() and (d / "images").is_dir()
    )


def _eval_yolo_custom(run_dir: Path, images_dir: Path, labels_dir: Path,
                      class_names: list, val_cfg: dict, tmp_root: Path) -> dict | None:
    from ultralytics import YOLO

    weights = run_dir / "weights" / "best.pt"
    if not weights.exists():
        return None

    tmp = tmp_root / run_dir.name
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    (tmp / "images").symlink_to(images_dir.resolve())
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

    meta = _load_meta(run_dir)
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
            "run":          run_dir.name,
            "framework":    "yolo",
            "timestamp":    meta.get("timestamp", ""),
            "batches":      ",".join(meta.get("batches", [])),
            "train_images": meta.get("train_images", ""),
            "val_images":   meta.get("val_images", ""),
            "test_images":  meta.get("test_images", ""),
            "mAP50":        round(float(metrics.box.map50), 4),
            "mAP50-95":     round(float(metrics.box.map),   4),
            "Precision":    round(float(metrics.box.mp),    4),
            "Recall":       round(float(metrics.box.mr),    4),
            "weights":      str(weights),
            "is_best":      False,
        }
    except Exception as e:
        print(f"    [warn] YOLO eval failed: {e}")
        return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _detect_custom_yolo(run_dir: Path, test_imgs: list, val_cfg: dict, out_dir: Path):
    from ultralytics import YOLO
    weights = run_dir / "weights" / "best.pt"
    if not weights.exists():
        return
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    model = YOLO(str(weights))
    conf  = val_cfg.get("conf", 0.5)
    imgsz = val_cfg.get("imgsz", 640)
    for img_path in test_imgs:
        result = model(str(img_path), conf=conf, imgsz=imgsz, verbose=False)[0]
        result.save(str(out_dir / img_path.name))
    print(f"      → {out_dir.name}/ ({len(test_imgs)} images)")


def _detect_custom_rfdetr(run_dir: Path, test_imgs: list, val_cfg: dict, class_names: list, out_dir: Path):
    import warnings
    import cv2
    import numpy as np
    import supervision as sv
    warnings.filterwarnings("ignore", category=FutureWarning, module="rfdetr")
    from rfdetr import RFDETR

    weights = run_dir / "rfdetr" / "checkpoint_best_total.pth"
    if not weights.exists():
        return
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

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
    print(f"      → {out_dir.name}/ ({len(test_imgs)} images)")


def _mark_best_per_group(results: list, group_key: str) -> list:
    groups = {}
    for r in results:
        groups.setdefault(r[group_key], []).append(r)
    for group_results in groups.values():
        best = max(
            (r for r in group_results if isinstance(r.get("mAP50"), float)),
            key=lambda r: r["mAP50"],
            default=None,
        )
        for r in group_results:
            r["is_best"] = bool(
                best and r["run"] == best["run"] and r["framework"] == best["framework"]
            )
    return results


def update_custom_blind_test_leaderboard(workspace_dir: Path, results: list) -> list:
    headers = [
        "test_set", "run", "framework", "timestamp", "batches", "train_images", "val_images", "test_images",
        "mAP50", "mAP50-95", "Precision", "Recall", "weights", "is_best",
    ]
    results = _mark_best_per_group(results, "test_set")
    _save_csv(results, workspace_dir / "custom_blind_test_leaderboard.csv", headers)
    return results


def phase_custom_blind_test(runs_dir: Path, datasets_dir: Path, val_cfg: dict,
                            workspace_dir: Path, class_names: list,
                            current_run: str | None = None) -> list:
    banner("Phase 6b — Custom Blind Test (user-supplied folders)")

    custom_dir = ensure_custom_blind_test_dir(datasets_dir)
    test_sets  = discover_custom_test_sets(custom_dir)
    if not test_sets:
        print(f"  No test-set folders in {custom_dir} — skipping.")
        return []

    run_dirs     = sorted(d for d in runs_dir.iterdir() if d.is_dir())
    total        = len(run_dirs)
    all_results  = []
    tmp_root     = workspace_dir / ".tmp_custom_blind_test"

    for test_set_dir in test_sets:
        test_set_name = test_set_dir.name
        images_dir    = test_set_dir / "images"
        labels_dir    = test_set_dir / "labels"

        print(f"\n  [{test_set_name}]")
        test_imgs = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in SUPPORTED_IMG)
        if not test_imgs:
            print(f"    No images in {images_dir} — skipping.")
            continue

        has_labels = labels_dir.is_dir() and any(labels_dir.iterdir())
        set_results = []

        if has_labels:
            for i, run_dir in enumerate(run_dirs, 1):
                if (run_dir / "weights" / "best.pt").exists():
                    print(f"    [{i}/{total}] {run_dir.name} [yolo] ...")
                    r = _eval_yolo_custom(run_dir, images_dir, labels_dir, class_names, val_cfg,
                                          tmp_root / test_set_name)
                    if r:
                        r["test_set"] = test_set_name
                        set_results.append(r)
            all_results.extend(set_results)
        else:
            print(f"    No labels in {labels_dir} — skipping metrics (visual overlays only).")

        print(f"    Generating detection overlays on {len(test_imgs)} images ...")
        for i, run_dir in enumerate(run_dirs, 1):
            print(f"    [{i}/{total}] {run_dir.name}")
            base_out = run_dir / "detect_custom_blind_test" / test_set_name
            if (run_dir / "weights" / "best.pt").exists():
                _detect_custom_yolo(run_dir, test_imgs, val_cfg, base_out / "yolo")
            if (run_dir / "rfdetr" / "checkpoint_best_total.pth").exists():
                _detect_custom_rfdetr(run_dir, test_imgs, val_cfg, class_names, base_out / "rfdetr")

        if set_results:
            _print_ranked_table(set_results, current_run, f"Custom Blind Test — {test_set_name}")

    shutil.rmtree(tmp_root, ignore_errors=True)

    if all_results:
        all_results = update_custom_blind_test_leaderboard(workspace_dir, all_results)
        print(f"\n  Saved → {workspace_dir / 'custom_blind_test_leaderboard.csv'}")

    return all_results


# ─── Phase 7: Print ranked summary ───────────────────────────────

def _fmt_metric(val, best_val):
    if not isinstance(val, float):
        return f"{'N/A':>8}"
    star = "*" if val == best_val else " "
    return f"{val:>7.4f}{star}"


def _print_ranked_table(results: list, current_run: str | None, label: str):
    if not results:
        return

    sorted_results = sorted(
        results,
        key=lambda r: r["mAP50"] if isinstance(r.get("mAP50"), float) else 0,
        reverse=True,
    )
    best_row  = sorted_results[0]
    MCOLS     = ["mAP50", "mAP50-95", "Precision", "Recall"]
    best_vals = {
        m: max((r[m] for r in results if isinstance(r.get(m), float)), default=None)
        for m in MCOLS
    }

    RUN_W = 34
    FRM_W = 6

    print(f"\n--- {label} ---")
    hdr = (f"{'Rank':<5}  {'Run':{RUN_W}}  {'Frm':{FRM_W}}  {'Batches':<18}  "
           f"{'mAP50':>8}  {'mAP50-95':>9}  {'Prec':>8}  {'Recall':>8}")
    print(hdr)
    print("-" * len(hdr))

    for rank, r in enumerate(sorted_results, 1):
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
    return best_row


def phase_summary(
    eval_results:       list,
    blind_test_results: list,
    current_run:        str | None,
    batches:            list,
    train_count:        int,
    val_count:          int,
    test_count:         int,
    workspace_dir:      Path,
):
    if not eval_results:
        print("\nNo evaluation results to summarize.")
        return

    batch_label = "+".join(b.name for b in batches) if batches else "—"
    banner(f"RETRAINING SUMMARY — {current_run or 'eval-only'}")
    print(f"  Batches  : {batch_label}")
    if batches:
        print(f"  Dataset  : {train_count} train / {val_count} val / {test_count} test images")
    print(f"  Val leaderboard  → {workspace_dir / 'leaderboard.csv'}")
    if blind_test_results:
        print(f"  Test leaderboard → {workspace_dir / 'blind_test_leaderboard.csv'}")

    best_val = _print_ranked_table(
        eval_results, current_run, "VALIDATION (labeled val set)"
    )
    best_bt = _print_ranked_table(
        blind_test_results, current_run,
        f"BLIND TEST (held-out 10% of val — {test_count} images)"
    )

    print(f"\n{'='*62}")
    if best_val:
        print(f"  Best val model   : {best_val['framework'].upper()}  mAP50={best_val['mAP50']}")
        print(f"  Val weights      : {best_val['weights']}")
    if best_bt:
        print(f"  Best blind test  : {best_bt['framework'].upper()}  mAP50={best_bt['mAP50']}")
        print(f"  Test weights     : {best_bt['weights']}")
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
        description="Iterative retraining: merge → train YOLO + RF-DETR → evaluate → blind test"
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
    train_cfg  = cfg.get("train",        {})
    rfdetr_cfg = cfg.get("rfdetr_train", {})
    val_cfg    = cfg.get("validate",     {})

    run_rfdetr = (
        not args.eval_only
        and not args.skip_rfdetr
        and rfdetr_cfg.get("enabled", True)
    )

    # Phase 1 — Scan batches
    banner("Phase 1 — Scanning dataset batches")
    batches = scan_batches(datasets_dir, expected_classes)
    print(f"  Found {len(batches)} batch(es): {[b.name for b in batches]}")

    # Phase 2 — Merge YOLO dataset (with 10% val → test split)
    banner("Phase 2 — Merging dataset (symlinks, 10% val → test)")
    merged_dir, train_count, val_count, test_count = merge_dataset(
        batches, workspace_dir, expected_classes
    )
    print(f"  {train_count} train / {val_count} val / {test_count} test images → {merged_dir}")

    # Phase 2b — Convert to COCO (for RF-DETR, includes test split)
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
            train_cfg, merged_dir, batches, train_count, val_count, test_count, runs_dir
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

    # Phase 4 — Evaluate all models (val set)
    eval_results = phase_evaluate_all(runs_dir, merged_dir, merged_coco_dir, val_cfg)
    if not eval_results:
        print("\n  No trained models found. Run without --eval-only first.")
        return

    # Phase 5 — Val leaderboard
    banner("Phase 5 — Updating val leaderboard")
    eval_results = update_leaderboard(workspace_dir, eval_results)
    print(f"  Saved → {workspace_dir / 'leaderboard.csv'}")

    # Phase 6 — Blind test
    blind_test_results = phase_blind_test_all(
        runs_dir, merged_dir, merged_coco_dir, val_cfg, workspace_dir, expected_classes
    )

    # Phase 6b — Custom blind test (user-supplied folders)
    phase_custom_blind_test(
        runs_dir, datasets_dir, val_cfg, workspace_dir, expected_classes, current_run
    )

    # Phase 7 — Summary
    phase_summary(
        eval_results, blind_test_results, current_run,
        batches, train_count, val_count, test_count, workspace_dir,
    )


if __name__ == "__main__":
    main()
