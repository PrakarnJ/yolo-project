"""Batch tab — run selected models over a folder of images, browse results."""
from pathlib import Path

import cv2
import gradio as gr

from .inference import (BASE_CONF, det_list_from, filter_preds, format_detections,
                        predict_raw, render_preds)
from .ui_shared import MAX_MODELS, make_output_columns, output_list, pack_results

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_IMAGES = 200
THUMB_W    = 320
SORTS      = ["detections ↓", "max conf ↓", "name"]


def _class_name(raw, c):
    return raw["class_names"][c] if c < len(raw["class_names"]) else str(c)


def _draw_thumb(item: dict, raw: dict, idx) -> "np.ndarray":
    """Annotate the cached thumbnail with scaled-down boxes. Returns RGB."""
    img = item["thumb"].copy()
    s   = item["scale"]
    for di in idx:
        x1, y1, x2, y2 = (raw["boxes_xyxy"][di] * s).astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (60, 200, 230), 1)
        cv2.putText(img, _class_name(raw, int(raw["cls_ids"][di])),
                    (x1, max(y1 - 2, 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (60, 200, 230), 1)
    return img[:, :, ::-1]


def _gallery_items(state: dict, conf: float, sort: str):
    """(gallery items, display order) — thumbs annotated with the first model."""
    key0 = state["keys"][0]
    scored = []
    for i, item in enumerate(state["items"]):
        raw = item["preds"][key0]
        idx = filter_preds(raw, conf)
        mc  = float(raw["confs"][idx].max()) if len(idx) else 0.0
        scored.append((i, len(idx), mc))

    if sort == "detections ↓":
        scored.sort(key=lambda t: (-t[1], -t[2]))
    elif sort == "max conf ↓":
        scored.sort(key=lambda t: (-t[2], -t[1]))
    else:
        scored.sort(key=lambda t: state["items"][t[0]]["name"])

    items, order = [], []
    for i, n, mc in scored:
        item = state["items"][i]
        raw  = item["preds"][key0]
        idx  = filter_preds(raw, conf)
        items.append((_draw_thumb(item, raw, idx),
                      f"{item['name']} — {n} det(s), max {mc:.2f}"))
        order.append(i)
    return items, order


def run_batch(folder, model_keys, conf, crop_mode, person_conf, sort,
              progress=gr.Progress()):
    if not (folder or "").strip() or not model_keys:
        return None, "Enter a folder path and select model(s) first.", []
    d = Path(folder.strip()).expanduser()
    if not d.is_dir():
        return None, f"❌ Not a folder: `{d}`", []

    paths = sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    warn = ""
    if len(paths) > MAX_IMAGES:
        warn  = f" ⚠️ Capped at first {MAX_IMAGES} of {len(paths)} images."
        paths = paths[:MAX_IMAGES]
    if not paths:
        return None, f"No images found in `{d}`.", []

    keys, items = model_keys[:MAX_MODELS], []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        s = THUMB_W / img.shape[1]
        items.append({"path": str(p), "name": p.name, "scale": s,
                      "thumb": cv2.resize(img, (THUMB_W, max(1, int(img.shape[0] * s)))),
                      "preds": {}})

    total, done = len(keys) * len(items), 0
    for key in keys:  # models outer → one GPU model swap per model, not per image
        for item in items:
            img = cv2.imread(item["path"])
            raw = predict_raw(img, key, BASE_CONF, crop_mode, person_conf,
                              src_path=item["path"])
            raw["crop_imgs"] = None  # not needed in batch; keep memory bounded
            item["preds"][key] = raw
            done += 1
            progress(done / total, desc=f"{key.split(' | ', 1)[-1]} — {item['name']}")

    state = {"items": items, "keys": keys}
    gallery, order = _gallery_items(state, conf, sort)
    state["order"] = order
    status = (f"Ran {len(keys)} model(s) × {len(items)} image(s).{warn} "
              "Conf slider and Sort re-rank from cache; click a thumbnail to inspect.")
    return state, status, gallery


def resort(state, conf, sort):
    if not state:
        return state, gr.update(), gr.update()
    gallery, order = _gallery_items(state, conf, sort)
    state["order"] = order
    return state, gallery, f"Re-ranked at conf ≥ {conf:.2f} (from cache)."


def show_detail(state, conf, evt: gr.SelectData):
    if not state or evt.index is None:
        return pack_results([])
    item = state["items"][state["order"][evt.index]]
    img  = cv2.imread(item["path"])
    results = []
    for key in state["keys"]:
        raw = item["preds"][key]
        idx = filter_preds(raw, conf)
        ann = render_preds(img, raw, idx)[:, :, ::-1]
        results.append((f"{key}<br>`{item['name']}`", ann,
                        format_detections(det_list_from(raw, idx))))
    return pack_results(results)


def build_batch_tab(shared_inputs: list):
    model_dd, conf_sl, crop_cb, p_conf_sl = shared_inputs
    with gr.Tab("Batch"):
        batch_state = gr.State(None)
        with gr.Row():
            folder_tb = gr.Textbox(label="Image folder path", scale=3,
                                   placeholder="e.g. datasets/ppe_stecon/stecon-poc-batch1/val/images")
            sort_dd   = gr.Dropdown(SORTS, value=SORTS[0], label="Sort", scale=1)
            batch_run = gr.Button("Run batch", variant="primary", scale=1)
        batch_status = gr.Markdown()
        gallery = gr.Gallery(columns=6, label="Results — annotated with first model",
                             object_fit="contain")
        gr.Markdown("### Selected image — all models")
        det_cols = make_output_columns("image")

        batch_run.click(
            fn=run_batch,
            inputs=[folder_tb, model_dd, conf_sl, crop_cb, p_conf_sl, sort_dd],
            outputs=[batch_state, batch_status, gallery],
        )
        conf_sl.release(fn=resort, inputs=[batch_state, conf_sl, sort_dd],
                        outputs=[batch_state, gallery, batch_status])
        sort_dd.change(fn=resort, inputs=[batch_state, conf_sl, sort_dd],
                       outputs=[batch_state, gallery, batch_status])
        gallery.select(fn=show_detail, inputs=[batch_state, conf_sl],
                       outputs=output_list(det_cols))
