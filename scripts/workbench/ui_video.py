"""Video tab — per-model annotated outputs plus detection timeline & frame scrubber.

Conf here is the inference threshold (unlike the Image tab): caching raw
predictions for every frame × model is memory-prohibitive, so scrubbing reads
back the already-annotated output videos instead of re-running any model.
"""
import tempfile
from collections import Counter

import cv2
import gradio as gr

from .inference import dispatch, format_detections, stitch_comparison
from .ui_shared import MAX_MODELS, empty_outputs, make_output_columns, output_list, pack_results


def _stitch_videos(results: list) -> str:
    """Read the per-model output videos in lockstep and write one side-by-side
    comparison video so all panels play in sync."""
    import imageio

    caps = [cv2.VideoCapture(path) for _, path, _ in results]
    labels = [key.split(" | ", 1)[-1] for key, _, _ in results]
    fps = caps[0].get(cv2.CAP_PROP_FPS) or 25

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out_path = tmp.name
    tmp.close()
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264",
                                output_params=["-pix_fmt", "yuv420p"])
    while True:
        frames = []
        for cap in caps:
            ret, frame_bgr = cap.read()
            if not ret:
                frames = None
                break
            frames.append(frame_bgr)
        if frames is None:
            break
        combined = stitch_comparison(list(zip(labels, frames)))
        writer.append_data(combined[:, :, ::-1])
    for cap in caps:
        cap.release()
    writer.close()
    return out_path


def _empty_extra():
    """Blank updates for (state, timeline plot, frame slider, preview image)."""
    return (None, gr.update(visible=False), gr.update(), gr.update(visible=False))


def handle_video(video_path, model_keys, conf, crop_mode, person_conf, stride,
                 progress=gr.Progress()):
    if video_path is None or not model_keys:
        return (*empty_outputs(), gr.update(visible=False), *_empty_extra())
    import imageio
    import pandas as pd

    stride = max(1, int(stride))
    results, timeline_rows, n_frames = [], [], 0
    keys = model_keys[:MAX_MODELS]
    for mi, key in enumerate(keys):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        out_path = tmp.name
        tmp.close()
        all_dets, frame_i = [], 0
        # only sampled frames are written; reduced fps keeps the duration intact
        writer = imageio.get_writer(out_path, fps=max(fps / stride, 1), codec="libx264",
                                    output_params=["-pix_fmt", "yuv420p"])
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            if frame_i % stride == 0:
                annotated_bgr, dets = dispatch(frame_bgr, key, conf, crop_mode, person_conf)
                all_dets.extend(dets)
                if mi == 0:  # timeline tracks the first selected model
                    for cls, cnt in Counter(d["class"] for d in dets).items():
                        timeline_rows.append({"frame": frame_i, "class": cls, "count": cnt})
                writer.append_data(annotated_bgr[:, :, ::-1])
            frame_i += 1
            if total:
                progress((mi + frame_i / total) / len(keys),
                         desc=f"{key.split(' | ', 1)[-1]} — frame {frame_i}/{total}"
                              + (f" (every {stride})" if stride > 1 else ""))
        cap.release()
        writer.close()
        n_frames = max(n_frames, frame_i)
        results.append((key, out_path, format_detections(all_dets)))

    if len(results) >= 2:
        combined = gr.update(value=_stitch_videos(results), visible=True)
    else:
        combined = gr.update(visible=False)

    det_frames = sorted({r["frame"] for r in timeline_rows})
    state = {"outs": [(key.split(" | ", 1)[-1], path) for key, path, _ in results],
             "n_frames": n_frames, "det_frames": det_frames, "stride": stride}

    if timeline_rows:
        plot_upd = gr.update(value=pd.DataFrame(timeline_rows), visible=True)
    else:
        plot_upd = gr.update(visible=False)
    # slider stays in SOURCE-frame units; step = stride so it snaps to sampled frames
    slider_upd = gr.update(maximum=max(n_frames - 1, 1), value=0, step=stride)
    preview    = _scrub_frame(state, 0)

    return (*pack_results(results), combined, state, plot_upd, slider_upd, preview)


def _scrub_frame(state, frame_i):
    if not state:
        return gr.update()
    # slider is in source-frame units; output videos only contain sampled frames
    out_idx = int(frame_i) // state.get("stride", 1)
    panels = []
    for label, path in state["outs"]:
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, out_idx)
        ret, frame = cap.read()
        cap.release()
        if ret:
            panels.append((label, frame))
    if not panels:
        return gr.update()
    return gr.update(value=stitch_comparison(panels)[:, :, ::-1], visible=True)


def scrub(state, frame_i):
    return _scrub_frame(state, frame_i)


def jump(state, cur, step):
    """Jump to the nearest frame with detections (wraps around)."""
    if not state or not state["det_frames"]:
        return gr.update(), gr.update()
    frames = state["det_frames"]
    if step > 0:
        target = next((f for f in frames if f > cur), frames[0])
    else:
        target = next((f for f in reversed(frames) if f < cur), frames[-1])
    return gr.update(value=target), _scrub_frame(state, target)


def build_video_tab(shared_inputs: list):
    with gr.Tab("Video"):
        vid_state   = gr.State(None)
        vid_in      = gr.Video(label="Input Video")
        with gr.Row():
            vid_run_btn = gr.Button("Run Inference", variant="primary", scale=3)
            stride_sl   = gr.Slider(1, 30, value=1, step=1, scale=2,
                                    label="Process every Nth frame",
                                    info="1 = all frames; higher = faster runs, "
                                         "output fps reduced to keep duration")
        gr.Markdown("*Confidence applies at inference time on this tab — re-run "
                    "after changing it. The scrubber below reads the annotated "
                    "outputs, no re-inference.*")
        vid_cmp     = gr.Video(label="Side-by-side comparison (synced)",
                               visible=False)
        vid_cols    = make_output_columns("video")

        with gr.Accordion("Detection timeline & frame scrubber", open=True):
            tl_plot = gr.LinePlot(x="frame", y="count", color="class",
                                  label="Detections per frame (first model)",
                                  visible=False)
            with gr.Row():
                prev_btn = gr.Button("⏮ Prev detection", scale=1)
                frame_sl = gr.Slider(0, 1, value=0, step=1, label="Frame", scale=4)
                next_btn = gr.Button("Next detection ⏭", scale=1)
            scrub_img = gr.Image(label="Frame preview", visible=False)

        vid_run_btn.click(
            fn=handle_video,
            inputs=[vid_in] + shared_inputs + [stride_sl],
            outputs=output_list(vid_cols) + [vid_cmp, vid_state, tl_plot,
                                             frame_sl, scrub_img],
        )
        frame_sl.release(fn=scrub, inputs=[vid_state, frame_sl], outputs=[scrub_img])
        prev_btn.click(fn=lambda s, c: jump(s, c, -1), inputs=[vid_state, frame_sl],
                       outputs=[frame_sl, scrub_img])
        next_btn.click(fn=lambda s, c: jump(s, c, +1), inputs=[vid_state, frame_sl],
                       outputs=[frame_sl, scrub_img])
