"""Top-level Gradio app assembly."""
import gradio as gr

from .ui_batch import build_batch_tab
from .ui_camera import build_camera_tab
from .ui_image import build_image_tab
from .ui_shared import build_header
from .ui_video import build_video_tab


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Detection Workbench") as demo:
        gr.Markdown("## Object Detection Workbench — YOLO & RF-DETR")

        fw_radio, model_dd, conf_sl, crop_cb, p_conf_sl = build_header()
        shared_inputs = [model_dd, conf_sl, crop_cb, p_conf_sl]

        with gr.Tabs():
            build_image_tab(shared_inputs)
            build_batch_tab(shared_inputs)
            build_video_tab(shared_inputs)
            build_camera_tab(shared_inputs)

    return demo
