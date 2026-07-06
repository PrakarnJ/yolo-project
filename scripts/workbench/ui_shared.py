"""Shared UI builders: header controls and the 4-slot output column layout."""
import gradio as gr

from .registry import MODEL_REGISTRY, get_run_info, register_custom_weight

MAX_MODELS = 4


def empty_outputs():
    """Return a blank state for all MAX_MODELS output slots."""
    return (
        *[gr.update(visible=False)] * MAX_MODELS,
        *[""] * MAX_MODELS,
        *[None] * MAX_MODELS,
        *[""] * MAX_MODELS,
    )


def pack_results(results: list):
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


def filter_models(category: str):
    if category == "YOLO":
        choices = [k for k in MODEL_REGISTRY if k.endswith("| YOLO")]
    elif category == "RF-DETR":
        choices = [k for k in MODEL_REGISTRY if k.endswith("| RF-DETR")]
    elif category == "Pose":
        choices = [k for k in MODEL_REGISTRY if k.endswith("| YOLO-Pose")]
    else:
        choices = list(MODEL_REGISTRY.keys())
    return gr.Dropdown(choices=choices, value=[choices[0]] if choices else [])


def make_output_columns(media_type: str):
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


def output_list(cols):
    """Flatten (col, lbl, media, stats) list into the outputs= order."""
    return (
        [c  for c, _, _, _ in cols] +
        [l  for _, l, _, _ in cols] +
        [m  for _, _, m, _ in cols] +
        [s  for _, _, _, s in cols]
    )


def load_custom_weight(path_str, class_override, selection):
    if not (path_str or "").strip():
        return gr.update(), "Enter a weight path first."
    try:
        label, msg = register_custom_weight(path_str, class_override or "")
    except Exception as e:
        return gr.update(), f"❌ {e}"
    selection = list(selection or [])
    if label not in selection:
        selection.append(label)
    return gr.Dropdown(choices=list(MODEL_REGISTRY.keys()), value=selection), msg


def show_model_details(keys):
    if not keys:
        return "Select model(s) to see run metadata."
    return "\n\n---\n\n".join(get_run_info(k) for k in keys)


def build_header():
    """Top control row shared by all tabs. Returns the individual components."""
    model_choices = list(MODEL_REGISTRY.keys())
    with gr.Row():
        fw_radio  = gr.Radio(["All", "YOLO", "RF-DETR", "Pose"], value="All",
                             label="Framework", scale=1)
        model_dd  = gr.Dropdown(choices=model_choices,
                                value=[model_choices[0]] if model_choices else [],
                                multiselect=True,
                                label="Model(s) — select multiple to compare", scale=3)
        conf_sl   = gr.Slider(0.05, 1.0, value=0.5, step=0.05, label="Confidence", scale=1)
        crop_cb   = gr.Checkbox(value=True, label="Crop by person first", scale=1)
        p_conf_sl = gr.Slider(0.1, 1.0, value=0.3, step=0.05, label="Person Conf", scale=1)

    with gr.Accordion("Load custom weights (.pt / .pth)", open=False):
        with gr.Row():
            w_path = gr.Textbox(label="Weight path", scale=3,
                                placeholder="/path/to/best.pt — YOLO classes auto-read from checkpoint")
            w_cls  = gr.Textbox(label="Class names override (comma-separated, optional)",
                                scale=2, placeholder="required for RF-DETR .pth")
            w_btn  = gr.Button("Load", scale=1)
        w_status = gr.Markdown()

    with gr.Accordion("Model details", open=False):
        details_md = gr.Markdown(show_model_details(model_choices[:1]))

    fw_radio.change(fn=filter_models, inputs=[fw_radio], outputs=[model_dd])
    w_btn.click(fn=load_custom_weight, inputs=[w_path, w_cls, model_dd],
                outputs=[model_dd, w_status])
    model_dd.change(fn=show_model_details, inputs=[model_dd], outputs=[details_md])
    return fw_radio, model_dd, conf_sl, crop_cb, p_conf_sl
