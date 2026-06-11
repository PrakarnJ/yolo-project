"""
Iterative retraining pipeline.

Workflow:
  1. Scan datasets/  — find batch folders, verify class consistency
  2. Merge           — symlink all batches into workspace/merged/
  3. Train           — train new model on merged dataset (skipped with --eval-only)
  4. Evaluate ALL    — val every workspace/runs/*/weights/best.pt on current val set
  5. Leaderboard     — update workspace/leaderboard.csv
  6. Detect          — run all models on test image folder, save annotated images
  7. Summary         — ranked table (validation + detection), show best model

Usage:
  .venv/bin/python scripts/retrain.py --config configs/retrain/ppe.yaml
  .venv/bin/python scripts/retrain.py --config configs/retrain/ppe.yaml --eval-only
  .venv/bin/python scripts/retrain.py --config configs/retrain/ppe.yaml --epochs 100
"""

import argparse
import csv
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


# ─── Phase 2: Merge dataset using symlinks ───────────────────────

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


# ─── Phase 3: Train new model ────────────────────────────────────

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


def phase_train(train_cfg: dict, merged_dir: Path, batches: list,
                train_count: int, val_count: int, runs_dir: Path) -> tuple:
    from ultralytics import YOLO

    run_num = _next_run_number(runs_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{run_num:03d}_{timestamp}"

    banner(f"Phase 3 — Training: {run_name}")

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
        "run": run_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "batches": [b.name for b in batches],
        "train_images": train_count,
        "val_images": val_count,
    }
    with open(run_dir / "run_meta.yaml", "w") as f:
        yaml.dump(meta, f)

    return run_name, run_dir


# ─── Phase 4: Evaluate all models ────────────────────────────────

def _load_meta(run_dir: Path) -> dict:
    meta_path = run_dir / "run_meta.yaml"
    if meta_path.exists():
        with open(meta_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def evaluate_model(run_dir: Path, merged_dir: Path, val_cfg: dict) -> dict | None:
    from ultralytics import YOLO

    weights = run_dir / "weights" / "best.pt"
    if not weights.exists():
        return None

    meta = _load_meta(run_dir)
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
        name="eval",
        verbose=False,
    )
    return {
        "run":          run_dir.name,
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


def phase_evaluate(runs_dir: Path, merged_dir: Path, val_cfg: dict) -> list:
    banner("Phase 4 — Evaluating all models")
    run_dirs = sorted(
        d for d in runs_dir.iterdir()
        if d.is_dir() and (d / "weights" / "best.pt").exists()
    )
    results = []
    for i, run_dir in enumerate(run_dirs, 1):
        print(f"  [{i}/{len(run_dirs)}] {run_dir.name} ...")
        r = evaluate_model(run_dir, merged_dir, val_cfg)
        if r:
            results.append(r)
    return results


# ─── Phase 5: Update leaderboard ─────────────────────────────────

def update_leaderboard(workspace_dir: Path, eval_results: list) -> list:
    lb_path = workspace_dir / "leaderboard.csv"
    headers = [
        "run", "timestamp", "batches", "train_images", "val_images",
        "mAP50", "mAP50-95", "Precision", "Recall", "weights", "is_best",
    ]
    if eval_results:
        best_run = max(eval_results, key=lambda r: r.get("mAP50") or 0)
        for r in eval_results:
            r["is_best"] = (r["run"] == best_run["run"])
    _save_csv(eval_results, lb_path, headers)
    return eval_results


# ─── Phase 6: Detect on image folder ─────────────────────────────

def detect_model(run_dir: Path, images: list, detect_cfg: dict) -> dict | None:
    from ultralytics import YOLO

    weights = run_dir / "weights" / "best.pt"
    if not weights.exists():
        return None

    conf  = detect_cfg.get("conf", 0.5)
    imgsz = detect_cfg.get("imgsz", 640)
    out_dir = run_dir / "detect"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    class_stats = defaultdict(lambda: {"count": 0, "conf_sum": 0.0, "img_names": set()})
    total_det = 0

    for img_path in images:
        result = model(str(img_path), conf=conf, imgsz=imgsz, verbose=False)[0]
        for box in result.boxes:
            cls_name = model.names[int(box.cls)]
            cs = float(box.conf)
            class_stats[cls_name]["count"] += 1
            class_stats[cls_name]["conf_sum"] += cs
            class_stats[cls_name]["img_names"].add(img_path.name)
            total_det += 1
        result.save(str(out_dir / img_path.name))

    images_with_det = len({n for s in class_stats.values() for n in s["img_names"]})
    top_class = max(class_stats, key=lambda c: class_stats[c]["count"]) if class_stats else "-"
    top_label = f"{top_class}({class_stats[top_class]['count']})" if class_stats else "-"

    meta = _load_meta(run_dir)
    return {
        "run":             run_dir.name,
        "timestamp":       meta.get("timestamp", ""),
        "total_det":       total_det,
        "images_with_det": images_with_det,
        "num_images":      len(images),
        "top_class":       top_label,
        "annotated_dir":   str(out_dir),
        "class_stats":     {
            cls: {
                "count":    s["count"],
                "avg_conf": round(s["conf_sum"] / s["count"], 3),
                "images":   len(s["img_names"]),
            }
            for cls, s in class_stats.items()
        },
    }


def phase_detect(runs_dir: Path, detect_cfg: dict, workspace_dir: Path) -> list:
    banner("Phase 6 — Detection on image folder")
    images_path = resolve(detect_cfg["images"])
    images = sorted(p for p in images_path.iterdir() if p.suffix.lower() in SUPPORTED_IMG)
    if not images:
        print(f"  No images found in {detect_cfg['images']} — skipping.")
        return []
    print(f"  Found {len(images)} images in {detect_cfg['images']}")

    run_dirs = sorted(
        d for d in runs_dir.iterdir()
        if d.is_dir() and (d / "weights" / "best.pt").exists()
    )
    det_results = []
    for i, run_dir in enumerate(run_dirs, 1):
        print(f"  [{i}/{len(run_dirs)}] {run_dir.name} ...")
        r = detect_model(run_dir, images, detect_cfg)
        if r:
            det_results.append(r)

    hist_path = workspace_dir / "detection_history.csv"
    _save_csv(
        [
            {
                "run":             r["run"],
                "timestamp":       r["timestamp"],
                "total_det":       r["total_det"],
                "images_with_det": r["images_with_det"],
                "num_images":      r["num_images"],
                "top_class":       r["top_class"],
                "annotated_dir":   r["annotated_dir"],
            }
            for r in det_results
        ],
        hist_path,
        ["run", "timestamp", "total_det", "images_with_det", "num_images", "top_class", "annotated_dir"],
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
    det_results: list,
    current_run: str | None,
    batches: list,
    train_count: int,
    val_count: int,
    detect_cfg: dict,
    workspace_dir: Path,
):
    if not eval_results:
        print("\nNo evaluation results to summarize.")
        return

    sorted_val = sorted(eval_results, key=lambda r: r.get("mAP50") or 0, reverse=True)
    best_val_run = sorted_val[0]["run"]
    MCOLS = ["mAP50", "mAP50-95", "Precision", "Recall"]
    best_metrics = {
        m: max((r[m] for r in eval_results if isinstance(r.get(m), float)), default=None)
        for m in MCOLS
    }

    batch_label = "+".join(b.name for b in batches) if batches else "—"
    header_run = current_run or "eval-only"
    banner(f"RETRAINING SUMMARY — {header_run}")
    print(f"  Batches  : {batch_label}")
    if batches:
        print(f"  Dataset  : {train_count} train images / {val_count} val images")
    if "images" in detect_cfg and det_results:
        n_img = det_results[0]["num_images"]
        print(f"  Test imgs: {detect_cfg['images']} ({n_img} images)")
    print(f"  Leaderboard → {workspace_dir / 'leaderboard.csv'}")

    def run_tag(run_name):
        parts = []
        if run_name == current_run:
            parts.append("NEW")
        if run_name == best_val_run:
            tag = "★"
        else:
            tag = " "
        suffix = f"  [{','.join(parts)}]" if parts else ""
        return tag, suffix

    RUN_W = 36
    print(f"\n--- VALIDATION (labeled val set) ---")
    hdr = (f"{'Rank':<5}  {'Run':{RUN_W}}  {'Batches':<22}  "
           f"{'mAP50':>8}  {'mAP50-95':>9}  {'Prec':>8}  {'Recall':>8}")
    print(hdr)
    print("-" * len(hdr))

    for rank, r in enumerate(sorted_val, 1):
        star, suffix = run_tag(r["run"])
        run_display = (star + " " + r["run"] + suffix)[:RUN_W]
        run_display = f"{run_display:{RUN_W}}"
        batch_str = r.get("batches", "")[:20]
        m50   = _fmt_metric(r.get("mAP50"),     best_metrics["mAP50"])
        m5095 = _fmt_metric(r.get("mAP50-95"),  best_metrics["mAP50-95"])
        prec  = _fmt_metric(r.get("Precision"),  best_metrics["Precision"])
        rec   = _fmt_metric(r.get("Recall"),     best_metrics["Recall"])
        print(f"{rank:<5}  {run_display}  {batch_str:<22}  {m50}  {m5095}  {prec}  {rec}")

    print(f"\n  ★ = this run   * = best across all runs")

    if det_results:
        sorted_det = sorted(det_results, key=lambda r: r["total_det"], reverse=True)
        best_det_run = sorted_det[0]["run"]
        print(f"\n--- DETECTION (image folder: {detect_cfg.get('images', '—')}) ---")
        hdr2 = (f"{'Rank':<5}  {'Run':{RUN_W}}  {'Total Det':>10}  "
                f"{'Imgs w/ Det':>12}  {'Top Class'}")
        print(hdr2)
        print("-" * len(hdr2))
        for rank, r in enumerate(sorted_det, 1):
            star2 = "★" if r["run"] == best_det_run else " "
            suffix2 = "  [NEW]" if r["run"] == current_run else ""
            run_display = (star2 + " " + r["run"] + suffix2)[:RUN_W]
            run_display = f"{run_display:{RUN_W}}"
            best_star = "*" if r["run"] == best_det_run else " "
            img_frac = f"{r['images_with_det']}/{r['num_images']}"
            print(f"{rank:<5}  {run_display}  {r['total_det']:>9}{best_star}  "
                  f"{img_frac:>12}  {r['top_class']}")
        print(f"\n  ★ = most detections this run   * = best total detections")

    best_row = sorted_val[0]
    print(f"\n{'='*62}")
    print(f"  Best model (mAP50={best_row['mAP50']}):")
    print(f"  {best_row['weights']}")
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
        description="Iterative retraining: merge batches → train → evaluate all → detect → leaderboard"
    )
    parser.add_argument("--config",    required=True, help="Path to retrain config YAML")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training; re-evaluate all existing models + update detection")
    parser.add_argument("--epochs",    type=int, help="Override epochs from config")
    args = parser.parse_args()

    with open(resolve(args.config)) as f:
        cfg = yaml.safe_load(f)

    if args.epochs:
        cfg.setdefault("train", {})["epochs"] = args.epochs

    datasets_dir = resolve(cfg.get("datasets_dir", "datasets"))
    workspace_dir = resolve(cfg.get("workspace_dir", "workspace"))
    runs_dir = workspace_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    expected_classes = cfg["classes"]
    train_cfg  = cfg.get("train",    {})
    val_cfg    = cfg.get("validate", {})
    detect_cfg = cfg.get("detect",   {})

    # Phase 1 — Scan batches
    banner("Phase 1 — Scanning dataset batches")
    batches = scan_batches(datasets_dir, expected_classes)
    print(f"  Found {len(batches)} batch(es): {[b.name for b in batches]}")

    # Phase 2 — Merge
    banner("Phase 2 — Merging dataset (symlinks)")
    merged_dir, train_count, val_count = merge_dataset(batches, workspace_dir, expected_classes)
    print(f"  {train_count} train images, {val_count} val images → {merged_dir}")

    # Phase 3 — Train
    current_run = None
    if not args.eval_only:
        current_run, _ = phase_train(train_cfg, merged_dir, batches, train_count, val_count, runs_dir)
    else:
        banner("Phase 3 — Skipped (--eval-only)")

    # Phase 4 — Evaluate all
    eval_results = phase_evaluate(runs_dir, merged_dir, val_cfg)
    if not eval_results:
        print("\n  No trained models found. Run without --eval-only first.")
        return

    # Phase 5 — Leaderboard
    banner("Phase 5 — Updating leaderboard")
    eval_results = update_leaderboard(workspace_dir, eval_results)
    print(f"  Saved → {workspace_dir / 'leaderboard.csv'}")

    # Phase 6 — Detect
    det_results = []
    if "images" in detect_cfg:
        det_results = phase_detect(runs_dir, detect_cfg, workspace_dir)
    else:
        banner("Phase 6 — Skipped (no detect.images in config)")

    # Phase 7 — Summary
    phase_summary(
        eval_results, det_results, current_run,
        batches, train_count, val_count, detect_cfg, workspace_dir,
    )


if __name__ == "__main__":
    main()
