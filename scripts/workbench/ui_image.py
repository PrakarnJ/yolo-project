"""Image tab — threshold explorer over cached raw predictions, with GT overlay."""
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from .inference import (BASE_CONF, det_list_from, detection_table, diagnostics_line,
                        filter_preds, format_detections, predict_raw,
                        render_disagreement, render_gt_overlay, render_preds)
from .matching import cross_model_clusters, greedy_match, gt_report, parse_yolo_labels
from .ui_shared import MAX_MODELS

TABLE_HEADERS = ["class", "conf", "w×h", "area px²"]


def _pack(results: list):
    """results: list of (key, img_rgb, diag_md, stats_md, table_rows)."""
    n = len(results)
    col_vis = [gr.update(visible=(i < n)) for i in range(MAX_MODELS)]
    labels  = [f"**{results[i][0]}**" if i < n else "" for i in range(MAX_MODELS)]
    media   = [results[i][1] if i < n else None for i in range(MAX_MODELS)]
    diags   = [results[i][2] if i < n else "" for i in range(MAX_MODELS)]
    stats   = [results[i][3] if i < n else "" for i in range(MAX_MODELS)]
    tables  = [results[i][4] if i < n else [] for i in range(MAX_MODELS)]
    return (*col_vis, *labels, *media, *diags, *stats, *tables)


def _parse_gt(gt_file, gt_path, state):
    """Resolve GT source (upload wins over path). Returns (gt|None, warning|None)."""
    src = gt_file or (gt_path or "").strip()
    if not src or not state:
        return None, None
    p = Path(src).expanduser()
    if not p.exists():
        return None, f"⚠️ GT file not found: `{p}`"
    h, w = state["image_bgr"].shape[:2]
    try:
        return parse_yolo_labels(p, w, h), None
    except Exception as e:
        return None, f"⚠️ GT parse error: {e}"


def _render_all(state: dict, conf: float, gt=None, iou_thr: float = 0.5):
    frame = state["image_bgr"]
    results = []
    for key, raw in state["preds"].items():
        idx = filter_preds(raw, conf)
        if gt is not None:
            tp, fp, fn = greedy_match(raw["boxes_xyxy"][idx], raw["cls_ids"][idx],
                                      raw["confs"][idx], gt["boxes_xyxy"],
                                      gt["cls_ids"], iou_thr)
            img   = render_gt_overlay(frame, raw, idx, gt, tp, fp, fn)[:, :, ::-1]
            stats = (gt_report(tp, fp, fn, raw["cls_ids"][idx], gt["cls_ids"],
                               raw["class_names"])
                     + "\n\n" + format_detections(det_list_from(raw, idx)))
        else:
            img   = render_preds(frame, raw, idx)[:, :, ::-1]
            stats = format_detections(det_list_from(raw, idx))
        results.append((key, img, diagnostics_line(raw), stats,
                        detection_table(raw, idx)))
    return _pack(results)


def _disagreement(state: dict, conf: float):
    """Cross-model comparison → (image update, markdown update). Hidden when <2 models."""
    preds = state["preds"]
    if len(preds) < 2:
        return gr.update(visible=False), gr.update(visible=False)

    model_dets = {}
    for key, raw in preds.items():
        idx   = filter_preds(raw, conf)
        names = [raw["class_names"][c] if c < len(raw["class_names"]) else str(c)
                 for c in raw["cls_ids"][idx]]
        model_dets[key] = (raw["boxes_xyxy"][idx], names, raw["confs"][idx])

    clusters = cross_model_clusters(model_dets)
    n   = len(preds)
    img = render_disagreement(state["image_bgr"], clusters, n)[:, :, ::-1]

    keys      = list(preds.keys())
    agreed    = sum(1 for c in clusters if len(c["models"]) == n)
    disputed  = [c for c in clusters if len(c["models"]) < n]
    lines = [f"**{agreed}** detection(s) found by all {n} models, "
             f"**{len(disputed)}** disputed:", ""]
    lines += [f"- M{i + 1} = {k}" for i, k in enumerate(keys)]
    if disputed:
        lines += ["", "| Class | " + " | ".join(f"M{i + 1}" for i in range(n)) + " | Found by |",
                  "|-------|" + "----|" * n + "----------|"]
        for c in disputed:
            confs = [f"{c['models'][k]:.2f}" if k in c["models"] else "—" for k in keys]
            found = ", ".join(f"M{keys.index(k) + 1}" for k in c["models"])
            lines.append(f"| {c['class']} | " + " | ".join(confs) + f" | {found} |")
    md = "\n".join(lines)
    return gr.update(value=img, visible=True), gr.update(value=md, visible=True)


def _crop_debug(state: dict, conf: float, show: bool):
    """Gallery of person crops (raw + annotated with the first model's detections)."""
    if not show or not state:
        return gr.update(visible=False), gr.update(visible=False)
    key, raw = next(iter(state["preds"].items()))
    if raw.get("crop_boxes") is None:
        return (gr.update(visible=False),
                gr.update(value="Crop mode was off for the last run.", visible=True))
    if not raw["crop_boxes"]:
        return (gr.update(visible=False),
                gr.update(value="⚠️ **Person detector found 0 people** — the model "
                                "ran on the full frame instead. If the scene does "
                                "contain people, lower Person Conf or debug "
                                "`yolo11n` on this input.", visible=True))

    idx   = filter_preds(raw, conf)
    items = []
    for i, ((x1, y1, _, _), crop) in enumerate(zip(raw["crop_boxes"], raw["crop_imgs"])):
        items.append((crop[:, :, ::-1], f"crop {i} — raw"))
        ann = crop.copy()
        sel = idx[raw["det_crop"][idx] == i]
        for di in sel:
            bx1, by1, bx2, by2 = raw["boxes_xyxy"][di]
            p1 = (int(bx1 - x1), int(by1 - y1))
            p2 = (int(bx2 - x1), int(by2 - y1))
            name = raw["class_names"][raw["cls_ids"][di]] \
                if raw["cls_ids"][di] < len(raw["class_names"]) else str(raw["cls_ids"][di])
            cv2.rectangle(ann, p1, p2, (60, 200, 230), 2)
            cv2.putText(ann, f"{name} {raw['confs'][di]:.2f}",
                        (p1[0], max(p1[1] - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 200, 230), 1)
        items.append((ann[:, :, ::-1], f"crop {i} — {len(sel)} det(s) [{key}]"))
    note = (f"{len(raw['crop_boxes'])} person crop(s); annotations from **{key}** "
            f"at conf ≥ {conf:.2f}.")
    return gr.update(value=items, visible=True), gr.update(value=note, visible=True)


def run_image(image_rgb, model_keys, conf, crop_mode, person_conf,
              gt_file, gt_path, iou_thr, show_crop_dbg):
    if image_rgb is None or not model_keys:
        return (None, "Upload an image and select model(s) first.", *_pack([]),
                gr.update(visible=False), gr.update(visible=False),
                gr.update(visible=False), gr.update(visible=False))
    frame_bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])
    preds = {key: predict_raw(frame_bgr, key, BASE_CONF, crop_mode, person_conf)
             for key in model_keys[:MAX_MODELS]}
    state = {"image_bgr": frame_bgr, "preds": preds}
    gt, warn = _parse_gt(gt_file, gt_path, state)
    status = (warn or f"Cached raw predictions from {len(preds)} model(s) at "
              f"conf ≥ {BASE_CONF} — the Confidence slider now re-filters "
              "instantly, no re-inference.")
    return (state, status, *_render_all(state, conf, gt, iou_thr),
            *_disagreement(state, conf), *_crop_debug(state, conf, show_crop_dbg))


def refilter(state, conf, gt_file, gt_path, iou_thr, show_crop_dbg):
    """Slider release / GT change → re-draw from cache. No model call."""
    if not state:
        return gr.update(), *(gr.update() for _ in range(6 * MAX_MODELS + 4))
    gt, warn = _parse_gt(gt_file, gt_path, state)
    status = warn or f"Re-filtered at conf ≥ {conf:.2f} from cache."
    return (status, *_render_all(state, conf, gt, iou_thr),
            *_disagreement(state, conf), *_crop_debug(state, conf, show_crop_dbg))


def invalidate():
    return None, "⚠️ Settings changed — press **Run Inference** to refresh the cache."


def build_image_tab(shared_inputs: list):
    model_dd, conf_sl, crop_cb, p_conf_sl = shared_inputs
    with gr.Tab("Image"):
        img_state = gr.State(None)
        img_in    = gr.Image(type="numpy", label="Input Image")

        with gr.Accordion("Ground truth (YOLO-format .txt — class ids must match "
                          "the model's class order)", open=False):
            with gr.Row():
                gt_file = gr.File(label="Upload label file", file_types=[".txt"],
                                  type="filepath", scale=2)
                gt_path = gr.Textbox(label="…or path to label file (press Enter)",
                                     scale=2)
                iou_sl  = gr.Slider(0.1, 0.9, value=0.5, step=0.05,
                                    label="Match IoU", scale=1)

        with gr.Row():
            img_run     = gr.Button("Run Inference", variant="primary", scale=4)
            crop_dbg_cb = gr.Checkbox(value=False, label="Show crop debug", scale=1)
        img_status = gr.Markdown()

        cols = []
        with gr.Row():
            for i in range(MAX_MODELS):
                with gr.Column(visible=(i == 0)) as col:
                    lbl   = gr.Markdown("")
                    media = gr.Image(label="Output", show_label=False)
                    diag  = gr.Markdown()
                    stats = gr.Markdown()
                    table = gr.Dataframe(headers=TABLE_HEADERS, interactive=False,
                                         label="Per-detection (≥ conf)")
                cols.append((col, lbl, media, diag, stats, table))

        with gr.Accordion("Model disagreement (select ≥2 models)", open=True):
            dis_img = gr.Image(show_label=False, visible=False)
            dis_md  = gr.Markdown(visible=False)

        with gr.Accordion("Crop pipeline debug", open=True):
            crop_note    = gr.Markdown(visible=False)
            crop_gallery = gr.Gallery(columns=4, visible=False, label="Person crops",
                                      object_fit="contain")

        outs = ([c for c, _, _, _, _, _ in cols] +
                [l for _, l, _, _, _, _ in cols] +
                [m for _, _, m, _, _, _ in cols] +
                [d for _, _, _, d, _, _ in cols] +
                [s for _, _, _, _, s, _ in cols] +
                [t for _, _, _, _, _, t in cols])

        gt_inputs = [gt_file, gt_path, iou_sl]
        img_run.click(
            fn=run_image,
            inputs=[img_in, model_dd, conf_sl, crop_cb, p_conf_sl] + gt_inputs
                   + [crop_dbg_cb],
            outputs=[img_state, img_status] + outs
                    + [dis_img, dis_md, crop_gallery, crop_note],
        )
        refilter_inputs  = [img_state, conf_sl] + gt_inputs + [crop_dbg_cb]
        refilter_outputs = [img_status] + outs + [dis_img, dis_md, crop_gallery, crop_note]
        conf_sl.release(fn=refilter, inputs=refilter_inputs, outputs=refilter_outputs)
        iou_sl.release(fn=refilter, inputs=refilter_inputs, outputs=refilter_outputs)
        gt_file.change(fn=refilter, inputs=refilter_inputs, outputs=refilter_outputs)
        gt_path.submit(fn=refilter, inputs=refilter_inputs, outputs=refilter_outputs)
        crop_dbg_cb.change(fn=refilter, inputs=refilter_inputs, outputs=refilter_outputs)

        for comp in (img_in, model_dd, crop_cb):
            comp.change(fn=invalidate, inputs=[], outputs=[img_state, img_status])
        p_conf_sl.release(fn=invalidate, inputs=[], outputs=[img_state, img_status])
