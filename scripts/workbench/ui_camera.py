"""Camera tab (live streaming)."""
import gradio as gr

from .inference import dispatch, format_detections, stitch_comparison


def handle_stream(frame_rgb, model_keys, conf, crop_mode, person_conf):
    if frame_rgb is None or not model_keys:
        return frame_rgb, ""
    frame_bgr = frame_rgb[:, :, ::-1]
    if len(model_keys) == 1:
        annotated_bgr, dets = dispatch(frame_bgr, model_keys[0], conf, crop_mode, person_conf)
        return annotated_bgr[:, :, ::-1], format_detections(dets)
    panels = []
    for key in model_keys:
        annotated_bgr, _ = dispatch(frame_bgr, key, conf, crop_mode, person_conf)
        panels.append((key.split(" | ", 1)[-1], annotated_bgr))
    return stitch_comparison(panels)[:, :, ::-1], ""


def build_camera_tab(shared_inputs: list):
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
