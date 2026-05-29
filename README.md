# PPE Detection — YOLO Project

YOLO11-based PPE (Personal Protective Equipment) detection for industrial safety monitoring (EGAT/Stecon).

**11 detection classes:** `helmet` · `Longsleeve` · `shortsleeve` · `coverall` · `Longpant` · `glove` · `shortpant` · `vest` · `skirt` · `boot` · `shoe`

---

## Project Structure

```
arv-yolo-project/
├── configs/                              # experiment configs (one file per run)
│   ├── train_ppe_stecon.yaml             # training hyperparams
│   ├── val_ppe_stecon.yaml               # validation settings
│   └── detect_ppe_stecon.yaml           # detection model + threshold
├── scripts/
│   ├── detect/
│   │   ├── detect_image.py              # detect on a single image
│   │   ├── detect_video.py              # detect on a video file
│   │   └── detect_webcam.py             # live detection from webcam
│   ├── train/
│   │   └── train_model.py               # fine-tune YOLO model
│   └── val/
│       └── val_model.py                 # evaluate model metrics
├── data/
│   ├── dataset/
│   │   ├── ppe_stecon_training/         # combined training dataset
│   │   ├── ppe-labeled_old/             # original labeled PPE data
│   │   └── egat_uat_newest_crop_combined_2025_1_13/   # EGAT UAT images
│   └── images/                          # sample images for quick tests
├── models/
│   ├── yolov8n.pt                       # base YOLOv8 nano weight
│   ├── yolo11n.pt                       # base YOLO11 nano weight
│   └── ppeweight_uategat/
│       └── yolo11n_uat.pt               # pretrained UAT weight (fine-tune starting point)
└── runs/                                # all outputs (auto-generated, git-ignored)
    ├── train/ppe-stecon/weights/best.pt
    ├── val/ppe-stecon/
    └── detect_image_output.jpg
```

---

## Datasets

| Dataset | Train | Val | Test | Notes |
|---------|------:|----:|-----:|-------|
| `ppe_stecon_training` | 1,220 | 1,994 | 1,938 | Combined dataset used to train `ppe-stecon` weight |
| `ppe-labeled_old` | 220 | 56 | — | Original labeled PPE images |
| `egat_uat_newest_crop_combined_2025_1_13` | 9,043 | 1,938 | 1,938 | EGAT UAT cropped images |

> `ppe_stecon_training` = `ppe-labeled_old` + `egat_uat_newest_crop_combined_2025_1_13` combined.

---

## Model Lineage

```
models/ppeweight_uategat/yolo11n_uat.pt   ← pretrained on EGAT UAT data
        │
        └── fine-tune on ppe_stecon_training (1000 epochs)
                │
                └── runs/train/ppe-stecon/weights/best.pt   ← ppe-stecon model
```

---

## Setup

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Requirements: Python 3.8+, CUDA-capable GPU.

---

## Config Files

All scripts are driven by config YAMLs. CLI flags override any config value.

**`configs/train_ppe_stecon.yaml`**
```yaml
model:    models/ppeweight_uategat/yolo11n_uat.pt
data:     data/dataset/ppe_stecon_training/ppe_stecon_training.yaml
epochs:   1000
imgsz:    720
batch:    16
freeze:   10
lr0:      0.001
patience: 100
project:  runs/train
name:     ppe-stecon
```

**`configs/val_ppe_stecon.yaml`**
```yaml
model:   runs/train/ppe-stecon/weights/best.pt
data:    data/dataset/ppe_stecon_training/ppe_stecon_training.yaml
split:   val
imgsz:   720
batch:   16
conf:    0.5
iou:     0.5
project: runs/val
name:    ppe-stecon
```

**`configs/detect_ppe_stecon.yaml`**
```yaml
model: runs/train/ppe-stecon/weights/best.pt
conf:  0.5
```

---

## Usage

### Training

```bash
# Train with config
python scripts/train/train_model.py --config configs/train_ppe_stecon.yaml

# Override specific params without editing the file
python scripts/train/train_model.py --config configs/train_ppe_stecon.yaml --epochs 100 --batch 8 --lr0 0.0005
```

Available overrides: `--model` `--data` `--epochs` `--imgsz` `--batch` `--freeze` `--lr0` `--patience` `--name`

### Validation

```bash
# Validate on val split
python scripts/val/val_model.py --config configs/val_ppe_stecon.yaml

# Validate on test split
python scripts/val/val_model.py --config configs/val_ppe_stecon.yaml --split test

# Validate with a different model
python scripts/val/val_model.py --config configs/val_ppe_stecon.yaml --model runs/train/my-run/weights/best.pt
```

Available overrides: `--model` `--data` `--split` `--imgsz` `--batch` `--conf` `--iou` `--name`

Output metrics: mAP50, mAP50-95, Precision, Recall.

### Detection

```bash
# Image — using config
python scripts/detect/detect_image.py data/images/safety1.jpg --config configs/detect_ppe_stecon.yaml

# Image — direct model path (no config needed)
python scripts/detect/detect_image.py data/images/safety1.jpg --model runs/train/ppe-stecon/weights/best.pt

# Video
python scripts/detect/detect_video.py data/videos/your_video.mp4 --config configs/detect_ppe_stecon.yaml

# Webcam (press Q to quit)
python scripts/detect/detect_webcam.py --config configs/detect_ppe_stecon.yaml
```

Available overrides: `--model` `--conf`

---

## Adding a New Experiment

1. Copy the relevant config and edit it:
   ```bash
   cp configs/train_ppe_stecon.yaml configs/train_my_experiment.yaml
   # update model, data, name, and hyperparams
   ```
2. Run:
   ```bash
   python scripts/train/train_model.py --config configs/train_my_experiment.yaml
   python scripts/val/val_model.py     --config configs/val_my_experiment.yaml
   ```

No script files need to be edited — only the config.
