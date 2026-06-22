import os
import tempfile
import warnings
from collections import defaultdict
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

WORKSPACE_CONFIGS = {
    "ppe":   (ROOT / "workspace_ppe",   ROOT / "configs/retrain/ppe.yaml"),
    "fight": (ROOT / "workspace_fight", ROOT / "configs/retrain/physical_fight.yaml"),
}

MAX_MODELS = 4


def _load_class_names(config_path: Path) -> list:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    names = []
    for item in cfg.get("classes", []):
        for part in str(item).split(","):
            part = part.strip()
            if part:
                names.append(part)
    return names


def discover_models() -> dict:
    registry = {}
    for ws_key, (ws_dir, cfg_path) in WORKSPACE_CONFIGS.items():
        runs_dir = ws_dir / "runs"
        if not runs_dir.exists():
            continue
        class_names = _load_class_names(cfg_path)
        for run_dir in sorted(d for d in runs_dir.iterdir() if d.is_dir()):
            yolo_w   = run_dir / "weights" / "best.pt"
            rfdetr_w = run_dir / "rfdetr"  / "checkpoint_best_total.pth"
            if yolo_w.exists():
                label = f"{ws_key} | {run_dir.name} | YOLO"
                registry[label] = {"framework": "yolo", "weights": str(yolo_w), "class_names": class_names}
            if rfdetr_w.exists():
                label = f"{ws_key} | {run_dir.name} | RF-DETR"
                registry[label] = {"framework": "rfdetr", "weights": str(rfdetr_w), "class_names": class_names}
    return registry


MODEL_REGISTRY = discover_models()


class ModelCache:
    """Keeps one task model loaded at a time to avoid exhausting GPU memory."""

    def __init__(self):
        self._key   = None
        self._model = None

    def get(self, key: str):
        if self._key == key:
            return self._model
        entry = MODEL_REGISTRY[key]
        self._model = None  # release old
        if entry["framework"] == "yolo":
            self._model = YOLO(entry["weights"])
        else:
            warnings.filterwarnings("ignore", category=FutureWarning, module="rfdetr")
            from rfdetr import RFDETR
            self._model = RFDETR.from_checkpoint(entry["weights"])
        self._key = key
        return self._model


class PersonDetector:
    """Lazy-loads the COCO-pretrained YOLO model for person detection (class 0)."""

    def __init__(self):
        self._model = None

    def _get(self):
        if self._model is None:
            self._model = YOLO(str(ROOT / "models/raw_weight/yolo11n.pt"))
        return self._model

    def detect(self, frame_bgr: np.ndarray, conf: float = 0.3) -> list:
        """Return list of [x1, y1, x2, y2] for every detected person."""
        result = self._get()(frame_bgr, classes=[0], conf=conf, verbose=False)[0]
        return [b.xyxy[0].tolist() for b in result.boxes]


_cache      = ModelCache()
_person_det = PersonDetector()


# ── Inference helpers ─────────────────────────────────────────────────────────

def format_detections(detections: list) -> str:
    if not detections:
        return "**No detections.**"

    agg = defaultdict(list)
    for d in detections:
        agg[d["class"]].append(d["confidence"])

    rows = []
    for cls in sorted(agg):
        confs = agg[cls]
        rows.append(
            f"| {cls} | {len(confs)} | {max(confs):.2f} | {sum(confs)/len(confs):.2f} |"
        )

    total_dets = len(detections)
    total_cls  = len(agg)
    return (
        "| Class | Count | Max Conf | Avg Conf |\n"
        "|-------|-------|----------|----------|\n"
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
    model = _cache.get(model_key)

    if entry["framework"] == "yolo":
        result   = model(frame_bgr, conf=conf, verbose=False)[0]
        names    = entry["class_names"]
        det_list = [
            {"class": names[int(b.cls)] if int(b.cls) < len(names) else str(int(b.cls)),
             "confidence": float(b.conf)}
            for b in result.boxes
        ]
        return result.plot(), det_list

    return _annotate_rfdetr(model, frame_bgr, entry["class_names"], conf)


def infer_with_person_crop(frame_bgr: np.ndarray, model_key: str, conf: float,
                            person_conf: float):
    persons = _person_det.detect(frame_bgr, conf=person_conf)
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


def _dispatch(frame_bgr: np.ndarray, model_key: str, conf: float,
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


# ── Gradio handlers ───────────────────────────────────────────────────────────

def _empty_outputs():
    """Return a blank state for all MAX_MODELS output slots."""
    return (
        *[gr.update(visible=False)] * MAX_MODELS,
        *[""] * MAX_MODELS,
        *[None] * MAX_MODELS,
        *[""] * MAX_MODELS,
    )


def _pack_results(results: list):
    """
    Pack a list of (key, image_rgb, stats_md) into the flat output tuple
    expected by the Image/Video button handlers.
    n = len(results), remaining slots are hidden.
    """
    n = len(results)
    col_vis = [gr.update(visible=(i < n)) for i in range(MAX_MODELS)]
    labels  = [f"**{results[i][0]}**" if i < n else "" for i in range(MAX_MODELS)]
    media   = [results[i][1]           if i < n else None for i in range(MAX_MODELS)]
    stats   = [results[i][2]           if i < n else "" for i in range(MAX_MODELS)]
    return (*col_vis, *labels, *media, *stats)


def handle_image(image_rgb, model_keys, conf, crop_mode, person_conf):
    if image_rgb is None or not model_keys:
        return _empty_outputs()

    frame_bgr = image_rgb[:, :, ::-1]
    results = []
    for key in model_keys[:MAX_MODELS]:
        annotated_bgr, dets = _dispatch(frame_bgr, key, conf, crop_mode, person_conf)
        results.append((key, annotated_bgr[:, :, ::-1], format_detections(dets)))

    return _pack_results(results)


def handle_video(video_path, model_keys, conf, crop_mode, person_conf):
    if video_path is None or not model_keys:
        return _empty_outputs()
    import imageio

    results = []
    for key in model_keys[:MAX_MODELS]:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        out_path = tmp.name
        tmp.close()
        all_dets = []
        writer = imageio.get_writer(out_path, fps=fps, codec="libx264",
                                    output_params=["-pix_fmt", "yuv420p"])
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            annotated_bgr, dets = _dispatch(frame_bgr, key, conf, crop_mode, person_conf)
            all_dets.extend(dets)
            writer.append_data(annotated_bgr[:, :, ::-1])
        cap.release()
        writer.close()
        results.append((key, out_path, format_detections(all_dets)))

    return _pack_results(results)


def handle_stream(frame_rgb, model_keys, conf, crop_mode, person_conf):
    if frame_rgb is None or not model_keys:
        return frame_rgb, ""
    frame_bgr = frame_rgb[:, :, ::-1]
    if len(model_keys) == 1:
        annotated_bgr, dets = _dispatch(frame_bgr, model_keys[0], conf, crop_mode, person_conf)
        return annotated_bgr[:, :, ::-1], format_detections(dets)
    panels = []
    for key in model_keys:
        annotated_bgr, _ = _dispatch(frame_bgr, key, conf, crop_mode, person_conf)
        panels.append((key.split(" | ", 1)[-1], annotated_bgr))
    return stitch_comparison(panels)[:, :, ::-1], ""


# ── UI ────────────────────────────────────────────────────────────────────────

model_choices = list(MODEL_REGISTRY.keys())


def filter_models(category: str):
    if category == "YOLO":
        choices = [k for k in MODEL_REGISTRY if k.endswith("| YOLO")]
    elif category == "RF-DETR":
        choices = [k for k in MODEL_REGISTRY if k.endswith("| RF-DETR")]
    else:
        choices = list(MODEL_REGISTRY.keys())
    return gr.Dropdown(choices=choices, value=[choices[0]] if choices else [])


def _make_output_columns(media_type: str):
    """
    Build MAX_MODELS output columns inside a gr.Row.
    Returns list of (column, label_md, media_component, stats_md).
    media_type: "image" | "video"
    """
    cols = []
    with gr.Row():
        for i in range(MAX_MODELS):
            with gr.Column(visible=(i == 0)) as col:
                lbl   = gr.Markdown("")
                if media_type == "image":
                    media = gr.Image(label="Output", show_label=False)
                else:
                    media = gr.Video(label="Output", show_label=False)
                stats = gr.Markdown()
            cols.append((col, lbl, media, stats))
    return cols


def _output_list(cols):
    """Flatten (col, lbl, media, stats) list into the outputs= order."""
    return (
        [c  for c, _, _, _ in cols] +
        [l  for _, l, _, _ in cols] +
        [m  for _, _, m, _ in cols] +
        [s  for _, _, _, s in cols]
    )


with gr.Blocks(title="Detection UI") as demo:
    gr.Markdown("## Object Detection — YOLO & RF-DETR")

    with gr.Row():
        fw_radio  = gr.Radio(["All", "YOLO", "RF-DETR"], value="All",
                             label="Framework", scale=1)
        model_dd  = gr.Dropdown(choices=model_choices,
                                value=[model_choices[0]] if model_choices else [],
                                multiselect=True,
                                label="Model(s) — select multiple to compare", scale=3)
        conf_sl   = gr.Slider(0.1, 1.0, value=0.5, step=0.05, label="Confidence", scale=1)
        crop_cb   = gr.Checkbox(value=True, label="Crop by person first", scale=1)
        p_conf_sl = gr.Slider(0.1, 1.0, value=0.3, step=0.05, label="Person Conf", scale=1)

    fw_radio.change(fn=filter_models, inputs=[fw_radio], outputs=[model_dd])

    shared_inputs = [model_dd, conf_sl, crop_cb, p_conf_sl]

    with gr.Tabs():
        with gr.Tab("Image"):
            img_in      = gr.Image(type="numpy", label="Input Image")
            img_run_btn = gr.Button("Run Inference")
            img_cols    = _make_output_columns("image")
            img_run_btn.click(
                fn=handle_image,
                inputs=[img_in] + shared_inputs,
                outputs=_output_list(img_cols),
            )

        with gr.Tab("Video"):
            vid_in      = gr.Video(label="Input Video")
            vid_run_btn = gr.Button("Run Inference")
            vid_cols    = _make_output_columns("video")
            vid_run_btn.click(
                fn=handle_video,
                inputs=[vid_in] + shared_inputs,
                outputs=_output_list(vid_cols),
            )

        with gr.Tab("Camera"):
            with gr.Row():
                cam_in    = gr.Image(sources=["webcam"], streaming=True,
                                     type="numpy", label="Camera")
                with gr.Column():
                    cam_out   = gr.Image(label="Live Output")
                    cam_stats = gr.Markdown(label="Detections")
            cam_in.stream(
                fn=handle_stream,
                inputs=[cam_in] + shared_inputs,
                outputs=[cam_out, cam_stats],
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
