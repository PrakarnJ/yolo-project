"""Inference, annotation, and frame-composition helpers.

Two inference paths coexist:
- predict_raw/filter_preds/render_preds — Image & Batch tabs. Runs once at
  BASE_CONF, caches raw arrays, re-filters/re-draws at any threshold with no
  further model calls.
- infer_with_detections/dispatch — Video & Camera tabs, where per-frame raw
  caching would be memory-prohibitive; conf is the inference threshold there.
"""
import os
import tempfile
import time
import warnings
from collections import defaultdict

import cv2
import numpy as np

from .registry import MODEL_REGISTRY, model_cache, person_detector

BASE_CONF = 0.05

# COCO 17-keypoint skeleton for pose redraw from cached arrays
_SKELETON = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8),
             (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


def _empty_raw():
    return {"boxes_xyxy": np.zeros((0, 4), np.float32), "confs": np.zeros(0, np.float32),
            "cls_ids": np.zeros(0, int), "kpts": None}


def _predict_single(frame_bgr: np.ndarray, model_key: str, base_conf: float,
                    src_path: str = None) -> dict:
    """One inference call → raw arrays dict (image coords local to frame_bgr).

    src_path: original on-disk image, if any — lets RF-DETR skip the temp-jpg
    round trip (used by the Batch tab).
    """
    entry = MODEL_REGISTRY[model_key]
    model = model_cache.get(model_key)
    t0 = time.perf_counter()

    if entry["framework"] in ("yolo", "yolo-pose"):
        result = model(frame_bgr, conf=base_conf, verbose=False)[0]
        raw = {
            "boxes_xyxy": result.boxes.xyxy.cpu().numpy().astype(np.float32),
            "confs":      result.boxes.conf.cpu().numpy().astype(np.float32),
            "cls_ids":    result.boxes.cls.cpu().numpy().astype(int),
            "kpts":       (result.keypoints.data.cpu().numpy()
                           if entry["framework"] == "yolo-pose" and result.keypoints is not None
                           else None),
        }
    else:  # rfdetr — file-based API
        warnings.filterwarnings("ignore", category=FutureWarning, module="rfdetr")
        if src_path:
            dets = model.predict(src_path, threshold=base_conf)
        else:
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            tmp_path = tmp.name
            tmp.close()
            try:
                cv2.imwrite(tmp_path, frame_bgr)
                dets = model.predict(tmp_path, threshold=base_conf)
            finally:
                os.unlink(tmp_path)
        if dets is None or len(dets) == 0:
            raw = _empty_raw()
        else:
            raw = {"boxes_xyxy": np.array(dets.xyxy, np.float32),
                   "confs":      np.array(dets.confidence, np.float32),
                   "cls_ids":    np.array(dets.class_id, int),
                   "kpts":       None}

    raw["timing_ms"] = (time.perf_counter() - t0) * 1000
    return raw


def _device_of(model_key: str) -> str:
    model = model_cache.get(model_key)
    try:
        return str(next(model.model.parameters()).device)
    except Exception:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"


def predict_raw(frame_bgr: np.ndarray, model_key: str, base_conf: float = BASE_CONF,
                crop_mode: bool = False, person_conf: float = 0.3,
                src_path: str = None) -> dict:
    """Run inference once at base_conf and return the full RawPred cache entry.

    Box coords are always FULL-IMAGE pixels: in crop mode, per-crop detections
    are offset back, so filtering/matching/rendering work identically either way.
    """
    entry = MODEL_REGISTRY[model_key]
    h, w  = frame_bgr.shape[:2]

    crop_boxes, crop_imgs = None, None
    if not crop_mode:
        raw = _predict_single(frame_bgr, model_key, base_conf, src_path)
        raw["det_crop"] = np.full(len(raw["confs"]), -1, int)
    else:
        persons = person_detector.detect(frame_bgr, conf=person_conf)
        crop_boxes, crop_imgs = [], []
        if not persons:
            raw = _predict_single(frame_bgr, model_key, base_conf)
            raw["det_crop"] = np.full(len(raw["confs"]), -1, int)
        else:
            parts, timing = [], 0.0
            for x1, y1, x2, y2 in persons:
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(w, int(x2)), min(h, int(y2))
                crop = frame_bgr[y1:y2, x1:x2]
                if crop.size == 0 or min(crop.shape[:2]) < 20:
                    continue
                part = _predict_single(crop, model_key, base_conf)
                part["boxes_xyxy"][:, [0, 2]] += x1
                part["boxes_xyxy"][:, [1, 3]] += y1
                if part["kpts"] is not None and len(part["kpts"]):
                    part["kpts"][:, :, 0] += x1
                    part["kpts"][:, :, 1] += y1
                timing += part["timing_ms"]
                crop_boxes.append([x1, y1, x2, y2])
                crop_imgs.append(crop.copy())
                parts.append(part)
            if parts:
                kpts_parts = [p["kpts"] for p in parts]
                raw = {
                    "boxes_xyxy": np.concatenate([p["boxes_xyxy"] for p in parts]),
                    "confs":      np.concatenate([p["confs"] for p in parts]),
                    "cls_ids":    np.concatenate([p["cls_ids"] for p in parts]),
                    "kpts":       (np.concatenate([k for k in kpts_parts if k is not None])
                                   if any(k is not None and len(k) for k in kpts_parts) else None),
                    "timing_ms":  timing,
                    "det_crop":   np.concatenate([np.full(len(p["confs"]), i, int)
                                                  for i, p in enumerate(parts)]),
                }
            else:
                raw = _empty_raw()
                raw["timing_ms"] = timing
                raw["det_crop"]  = np.zeros(0, int)

    raw["class_names"] = entry["class_names"]
    raw["crop_boxes"]  = crop_boxes
    raw["crop_imgs"]   = crop_imgs
    raw["imgsz"]       = f"{w}×{h}"
    raw["device"]      = _device_of(model_key)
    return raw


def filter_preds(raw: dict, conf: float) -> np.ndarray:
    """Indices of cached detections at/above the display threshold."""
    return np.where(raw["confs"] >= conf)[0]


def _class_name(raw: dict, cls_id: int) -> str:
    names = raw["class_names"]
    return names[cls_id] if cls_id < len(names) else str(cls_id)


def render_preds(frame_bgr: np.ndarray, raw: dict, idx: np.ndarray) -> np.ndarray:
    """Redraw annotations from cached arrays — no model involved."""
    import supervision as sv

    img = frame_bgr.copy()

    # person-crop context boxes first, so detections draw on top
    if raw.get("crop_boxes"):
        for i, (x1, y1, x2, y2) in enumerate(raw["crop_boxes"]):
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"person {i}", (x1, max(y1 - 6, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    if len(idx):
        sv_dets = sv.Detections(
            xyxy=raw["boxes_xyxy"][idx],
            confidence=raw["confs"][idx],
            class_id=raw["cls_ids"][idx],
        )
        labels = [f"{_class_name(raw, int(c))} {s:.2f}"
                  for c, s in zip(sv_dets.class_id, sv_dets.confidence)]
        img = sv.BoxAnnotator().annotate(scene=img, detections=sv_dets)
        img = sv.LabelAnnotator().annotate(scene=img, detections=sv_dets, labels=labels)

    # pose keypoints redrawn manually (cached arrays have no ultralytics Results)
    if raw.get("kpts") is not None:
        for i in idx:
            kp = raw["kpts"][i]
            vis = kp[:, 2] > 0.5
            for a, b in _SKELETON:
                if a < len(kp) and b < len(kp) and vis[a] and vis[b]:
                    cv2.line(img, (int(kp[a][0]), int(kp[a][1])),
                             (int(kp[b][0]), int(kp[b][1])), (255, 128, 0), 2)
            for x, y, v in kp:
                if v > 0.5:
                    cv2.circle(img, (int(x), int(y)), 3, (0, 128, 255), -1)

    return img


def render_gt_overlay(frame_bgr: np.ndarray, raw: dict, idx: np.ndarray,
                      gt: dict, tp: dict, fp: list, fn: list) -> np.ndarray:
    """GT-match rendering: TP green, FP red, missed GT (FN) blue. tp/fp indices
    are local positions within idx (as returned by greedy_match on the subset)."""
    img = frame_bgr.copy()

    if raw.get("crop_boxes"):
        for i, (x1, y1, x2, y2) in enumerate(raw["crop_boxes"]):
            cv2.rectangle(img, (x1, y1), (x2, y2), (128, 128, 128), 1)

    boxes, confs, cls = raw["boxes_xyxy"][idx], raw["confs"][idx], raw["cls_ids"][idx]
    for li in range(len(idx)):
        x1, y1, x2, y2 = map(int, boxes[li])
        ok    = li in tp
        color = (80, 200, 60) if ok else (60, 60, 230)          # BGR: green / red
        tag   = "" if ok else " FP"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{_class_name(raw, int(cls[li]))} {confs[li]:.2f}{tag}",
                    (x1, max(y1 - 6, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    for j in fn:
        x1, y1, x2, y2 = map(int, gt["boxes_xyxy"][j])
        cv2.rectangle(img, (x1, y1), (x2, y2), (230, 140, 30), 2)  # BGR: blue
        cv2.putText(img, f"MISS {_class_name(raw, int(gt['cls_ids'][j]))}",
                    (x1, min(y2 + 16, img.shape[0] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 140, 30), 2)
    return img


def render_disagreement(frame_bgr: np.ndarray, clusters: list, n_models: int) -> np.ndarray:
    """One box per detection cluster, colored by how many models found it:
    green = all, orange = some, red = only one."""
    img = frame_bgr.copy()
    for cl in clusters:
        k = len(cl["models"])
        if k == n_models:
            color = (80, 200, 60)     # BGR green
        elif k > 1:
            color = (0, 165, 255)     # orange
        else:
            color = (60, 60, 230)     # red
        x1, y1, x2, y2 = map(int, cl["box"])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{cl['class']} {k}/{n_models}", (x1, max(y1 - 6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img


def det_list_from(raw: dict, idx: np.ndarray) -> list:
    """Convert filtered cached detections into the dict list format_detections expects."""
    dets = []
    for i in idx:
        d = {"class": _class_name(raw, int(raw["cls_ids"][i])),
             "confidence": float(raw["confs"][i])}
        if raw.get("kpts") is not None:
            d["kpts_visible"] = int((raw["kpts"][i][:, 2] > 0.5).sum())
        dets.append(d)
    return dets


def detection_table(raw: dict, idx: np.ndarray) -> list:
    """Rows [class, conf, w×h, area] sorted by confidence desc, for gr.Dataframe."""
    order = idx[np.argsort(-raw["confs"][idx])] if len(idx) else idx
    rows = []
    for i in order:
        x1, y1, x2, y2 = raw["boxes_xyxy"][i]
        bw, bh = x2 - x1, y2 - y1
        rows.append([_class_name(raw, int(raw["cls_ids"][i])),
                     round(float(raw["confs"][i]), 3),
                     f"{int(bw)}×{int(bh)}", int(bw * bh)])
    return rows


def diagnostics_line(raw: dict) -> str:
    n_crops = len(raw["crop_boxes"]) if raw.get("crop_boxes") else 0
    crop_txt = f" | {n_crops} person crop(s)" if raw.get("crop_boxes") is not None else ""
    return (f"⏱ {raw['timing_ms']:.0f} ms | input {raw['imgsz']} | "
            f"{raw['device']} | {len(raw['confs'])} raw dets ≥ {BASE_CONF}{crop_txt}")


def format_detections(detections: list) -> str:
    if not detections:
        return "**No detections.**"

    agg  = defaultdict(list)
    kpts = defaultdict(list)
    for d in detections:
        agg[d["class"]].append(d["confidence"])
        if "kpts_visible" in d:
            kpts[d["class"]].append(d["kpts_visible"])

    header = "| Class | Count | Max Conf | Avg Conf |"
    sep    = "|-------|-------|----------|----------|"
    if kpts:
        header += " Avg Visible KPs |"
        sep    += "-----------------|"

    rows = []
    for cls in sorted(agg):
        confs = agg[cls]
        row = f"| {cls} | {len(confs)} | {max(confs):.2f} | {sum(confs)/len(confs):.2f} |"
        if kpts:
            k = kpts.get(cls)
            row += f" {sum(k)/len(k):.1f} |" if k else " – |"
        rows.append(row)

    total_dets = len(detections)
    total_cls  = len(agg)
    return (
        header + "\n" + sep + "\n"
        + "\n".join(rows)
        + f"\n\n**Total: {total_dets} detection(s) across {total_cls} class(es)**"
    )


def _annotate_rfdetr(model, frame_bgr: np.ndarray, class_names: list, conf: float):
    import supervision as sv

    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        cv2.imwrite(tmp_path, frame_bgr)
        warnings.filterwarnings("ignore", category=FutureWarning, module="rfdetr")
        dets = model.predict(tmp_path, threshold=conf)
    finally:
        os.unlink(tmp_path)

    if dets is None or len(dets) == 0:
        return frame_bgr, []

    sv_dets = sv.Detections(
        xyxy=np.array(dets.xyxy),
        confidence=np.array(dets.confidence),
        class_id=np.array(dets.class_id, dtype=int),
    )
    labels = [
        f"{class_names[int(c)] if int(c) < len(class_names) else int(c)} {s:.2f}"
        for c, s in zip(dets.class_id, dets.confidence)
    ]
    img = sv.BoxAnnotator().annotate(scene=frame_bgr.copy(), detections=sv_dets)
    img = sv.LabelAnnotator().annotate(scene=img, detections=sv_dets, labels=labels)

    det_list = [
        {"class": class_names[int(c)] if int(c) < len(class_names) else str(int(c)),
         "confidence": float(s)}
        for c, s in zip(dets.class_id, dets.confidence)
    ]
    return img, det_list


def infer_with_detections(frame_bgr: np.ndarray, model_key: str, conf: float):
    entry = MODEL_REGISTRY[model_key]
    model = model_cache.get(model_key)

    if entry["framework"] == "yolo":
        result   = model(frame_bgr, conf=conf, verbose=False)[0]
        names    = entry["class_names"]
        det_list = [
            {"class": names[int(b.cls)] if int(b.cls) < len(names) else str(int(b.cls)),
             "confidence": float(b.conf)}
            for b in result.boxes
        ]
        return result.plot(), det_list

    if entry["framework"] == "yolo-pose":
        result    = model(frame_bgr, conf=conf, verbose=False)[0]
        kpt_confs = result.keypoints.conf if result.keypoints is not None else None
        det_list  = []
        for i, b in enumerate(result.boxes):
            det = {"class": "person", "confidence": float(b.conf)}
            if kpt_confs is not None:
                det["kpts_visible"] = int((kpt_confs[i] > 0.5).sum())
            det_list.append(det)
        return result.plot(), det_list

    return _annotate_rfdetr(model, frame_bgr, entry["class_names"], conf)


def infer_with_person_crop(frame_bgr: np.ndarray, model_key: str, conf: float,
                            person_conf: float):
    persons = person_detector.detect(frame_bgr, conf=person_conf)
    if not persons:
        return infer_with_detections(frame_bgr, model_key, conf)

    output   = frame_bgr.copy()
    all_dets = []

    for x1, y1, x2, y2 in persons:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame_bgr.shape[1], x2), min(frame_bgr.shape[0], y2)
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0 or min(crop.shape[:2]) < 20:
            continue

        annotated_crop, dets = infer_with_detections(crop, model_key, conf)
        output[y1:y2, x1:x2] = annotated_crop
        all_dets.extend(dets)

        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(output, "person", (x1, max(y1 - 6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    return output, all_dets


def dispatch(frame_bgr: np.ndarray, model_key: str, conf: float,
             crop_mode: bool, person_conf: float):
    if crop_mode:
        return infer_with_person_crop(frame_bgr, model_key, conf, person_conf)
    return infer_with_detections(frame_bgr, model_key, conf)


def stitch_comparison(panels: list) -> np.ndarray:
    """Horizontally stack annotated frames with label bars — used for Camera streaming."""
    if not panels:
        return np.zeros((100, 400, 3), dtype=np.uint8)

    target_h = max(img.shape[0] for _, img in panels)
    bar_h = 32
    strips = []

    for label, img in panels:
        if img.shape[0] != target_h:
            scale = target_h / img.shape[0]
            img = cv2.resize(img, (int(img.shape[1] * scale), target_h))
        bar = np.zeros((bar_h, img.shape[1], 3), dtype=np.uint8)
        cv2.putText(bar, label, (6, bar_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        strips.append(np.vstack([bar, img]))

    return np.hstack(strips)
