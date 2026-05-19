# YOLO Object Detection Project

Real-time object detection using [YOLOv8](https://github.com/ultralytics/ultralytics) (nano model) with CUDA acceleration.

## Scripts

| Script | Description |
|--------|-------------|
| `scripts/detect_image.py` | Run detection on a single image |
| `scripts/detect_video.py` | Run detection on a video file |
| `scripts/detect_webcam.py` | Run live detection from webcam |

## Setup

```bash
pip install -r requirements.txt
```

Download the YOLOv8n model weights and place them at `models/yolov8n.pt`:

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
mv yolov8n.pt models/
```

## Usage

**Image detection:**
```bash
python scripts/detect_image.py data/images/harrypotter.jpg
```

**Video detection:**
```bash
python scripts/detect_video.py data/videos/your_video.mp4
```

**Webcam detection:**
```bash
python scripts/detect_webcam.py
# Press Q to quit
```

Output files are saved to the `runs/` directory.

## Requirements

- Python 3.8+
- CUDA-capable GPU (scripts use `model.to("cuda")`)
- See `requirements.txt` for Python package versions
