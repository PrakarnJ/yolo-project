"""IoU-based matching: ground-truth TP/FP/FN and cross-model disagreement."""
from collections import defaultdict
from pathlib import Path

import numpy as np


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N,4) and (M,4) xyxy boxes → (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix = np.maximum(0, np.minimum(a[:, None, 2], b[None, :, 2])
                    - np.maximum(a[:, None, 0], b[None, :, 0]))
    iy = np.maximum(0, np.minimum(a[:, None, 3], b[None, :, 3])
                    - np.maximum(a[:, None, 1], b[None, :, 1]))
    inter  = ix * iy
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union  = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def parse_yolo_labels(txt_path, img_w: int, img_h: int) -> dict:
    """YOLO label file (class cx cy w h, normalized) → pixel-space GT dict."""
    boxes, cls = [], []
    for line in Path(txt_path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        c = int(float(parts[0]))
        cx, cy, w, h = (float(v) for v in parts[1:5])
        boxes.append([(cx - w / 2) * img_w, (cy - h / 2) * img_h,
                      (cx + w / 2) * img_w, (cy + h / 2) * img_h])
        cls.append(c)
    return {"boxes_xyxy": np.array(boxes, np.float32).reshape(-1, 4),
            "cls_ids": np.array(cls, int)}


def greedy_match(pred_boxes, pred_cls, pred_confs, gt_boxes, gt_cls,
                 iou_thr: float = 0.5, class_aware: bool = True):
    """Match predictions (conf-desc order) to unclaimed same-class GT with IoU ≥ thr.

    Returns (tp: {pred_i: gt_j}, fp: [pred_i], fn: [gt_j]) — indices are local
    to the arrays passed in.
    """
    M = iou_matrix(pred_boxes, gt_boxes)
    claimed, tp = set(), {}
    for i in np.argsort(-pred_confs):
        best_j, best_iou = -1, iou_thr
        for j in range(len(gt_boxes)):
            if j in claimed:
                continue
            if class_aware and pred_cls[i] != gt_cls[j]:
                continue
            if M[i, j] >= best_iou:
                best_iou, best_j = M[i, j], j
        if best_j >= 0:
            tp[int(i)] = best_j
            claimed.add(best_j)
    fp = [int(i) for i in range(len(pred_boxes)) if int(i) not in tp]
    fn = [j for j in range(len(gt_boxes)) if j not in claimed]
    return tp, fp, fn


def gt_report(tp: dict, fp: list, fn: list, pred_cls, gt_cls, class_names: list) -> str:
    """Per-image markdown: overall P/R plus per-class TP/FP/FN table."""
    n_tp, n_fp, n_fn = len(tp), len(fp), len(fn)
    prec = n_tp / (n_tp + n_fp) if n_tp + n_fp else 0.0
    rec  = n_tp / (n_tp + n_fn) if n_tp + n_fn else 0.0

    def name(c):
        return class_names[c] if c < len(class_names) else str(c)

    per_cls = defaultdict(lambda: [0, 0, 0])  # cls → [tp, fp, fn]
    for i in tp:
        per_cls[name(int(pred_cls[i]))][0] += 1
    for i in fp:
        per_cls[name(int(pred_cls[i]))][1] += 1
    for j in fn:
        per_cls[name(int(gt_cls[j]))][2] += 1

    lines = [f"**GT match:** ✅ TP {n_tp} | ❌ FP {n_fp} | 🔍 FN (missed) {n_fn} — "
             f"**P {prec:.2f} / R {rec:.2f}**",
             "", "| Class | TP | FP | FN |", "|-------|----|----|----|"]
    for cls in sorted(per_cls):
        t, f, n = per_cls[cls]
        lines.append(f"| {cls} | {t} | {f} | {n} |")
    return "\n".join(lines)


def cross_model_clusters(model_dets: dict, iou_thr: float = 0.5) -> list:
    """Cluster detections across models by same class + IoU ≥ thr.

    model_dets: {model_key: (boxes_xyxy, cls_names_per_det, confs)} — class
    compared by NAME so models with different index orders still align.
    Returns clusters: {"class", "box", "models": {key: conf}} sorted so the
    most-disputed (fewest finders) come first.
    """
    flat = []
    for key, (boxes, names, confs) in model_dets.items():
        for b, n, c in zip(boxes, names, confs):
            flat.append({"key": key, "box": np.asarray(b, np.float32),
                         "cls": n, "conf": float(c)})
    flat.sort(key=lambda d: -d["conf"])

    clusters = []
    for det in flat:
        placed = False
        for cl in clusters:
            if cl["class"] != det["cls"] or det["key"] in cl["models"]:
                continue
            if iou_matrix(det["box"][None], cl["box"][None])[0, 0] >= iou_thr:
                cl["models"][det["key"]] = det["conf"]
                placed = True
                break
        if not placed:
            clusters.append({"class": det["cls"], "box": det["box"],
                             "models": {det["key"]: det["conf"]}})
    clusters.sort(key=lambda c: (len(c["models"]), -max(c["models"].values())))
    return clusters
