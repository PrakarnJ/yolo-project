# PPE Detection — YOLO Project

YOLO11-based PPE (Personal Protective Equipment) detection for industrial safety monitoring (EGAT/Stecon).

**11 detection classes:** `helmet` · `Longsleeve` · `shortsleeve` · `coverall` · `Longpant` · `glove` · `shortpant` · `vest` · `skirt` · `boot` · `shoe`

---

## Project Structure

```
yolo-project/
├── configs/                              # experiment configs (one file per run)
│   ├── train_ppe_stecon.yaml             # training hyperparams
│   ├── val_ppe_stecon.yaml               # single-model validation settings
│   ├── detect_ppe_stecon.yaml            # detection model + threshold
│   ├── compare_ppe_stecon.yaml           # multi-model validation comparison
│   └── compare_detect_group2.yaml        # multi-model detection on image folder
├── scripts/
│   ├── detect/
│   │   ├── image.py                     # detect on a single image
│   │   ├── video.py                     # detect on a video file
│   │   ├── webcam.py                    # live detection from webcam
│   │   └── compare_detect.py            # compare models on an image folder
│   ├── train/
│   │   └── model.py                     # fine-tune YOLO model
│   └── validate/
│       ├── model.py                     # evaluate single model metrics
│       └── compare_val.py               # compare multiple models on a dataset
├── data/
│   ├── dataset/
│   │   ├── ppe_stecon_training/         # combined training dataset
│   │   ├── ppe-labeled_old/             # original labeled PPE data
│   │   └── egat_uat_newest_crop_combined_2025_1_13/   # EGAT UAT images
│   └── images/                          # sample images for quick tests
├── models/
│   ├── egat.pt                          # trained EGAT weight
│   ├── stecon-1.pt                      # trained Stecon weight
│   ├── ppeweight_uategat/
│   │   └── yolo11n_uat.pt               # pretrained UAT weight (fine-tune starting point)
│   └── raw_weight/                      # raw/base weights
└── runs/                                # all outputs (auto-generated, git-ignored)
    ├── train/ppe-stecon/weights/best.pt
    ├── val/
    └── detect/
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
                ├── runs/train/ppe-stecon/weights/best.pt   ← best checkpoint
                └── runs/train/ppe-stecon/weights/last.pt   ← final epoch

models/egat.pt      ← trained EGAT weight
models/stecon-1.pt  ← trained Stecon weight
```

---

## Setup

```bash
python3 -m venv .venv
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

**`configs/compare_ppe_stecon.yaml`** — multi-model validation (requires labeled dataset)

```yaml
dataset: data/dataset/ppe_stecon_training/ppe_stecon_training.yaml
split:   val
imgsz:   720
batch:   16
conf:    0.5
iou:     0.5
project: runs/val

models:
  - name:    ppe-egat
    weights: models/egat.pt
  - name:    ppe-stecon
    weights: models/stecon-1.pt
```

**`configs/compare_detect_group2.yaml`** — multi-model detection on an image folder

```yaml
images:  data/images/group-2
conf:    0.5
imgsz:   720
project: runs/detect

models:
  - name:    ppe-egat
    weights: models/egat.pt
  - name:    ppe-stecon
    weights: models/stecon-1.pt
```

---

## Usage

### Training

```bash
# Train with config
python scripts/train/model.py --config configs/train_ppe_stecon.yaml

# Override specific params without editing the file
python scripts/train/model.py --config configs/train_ppe_stecon.yaml --epochs 100 --batch 8 --lr0 0.0005
```

Available overrides: `--model` `--data` `--epochs` `--imgsz` `--batch` `--freeze` `--lr0` `--patience` `--name`

### Validation

```bash
# Validate single model on val split
python scripts/validate/model.py --config configs/val_ppe_stecon.yaml

# Validate on test split
python scripts/validate/model.py --config configs/val_ppe_stecon.yaml --split test

# Validate with a different model
python scripts/validate/model.py --config configs/val_ppe_stecon.yaml --model runs/train/my-run/weights/best.pt
```

Available overrides: `--model` `--data` `--split` `--imgsz` `--batch` `--conf` `--iou` `--name`

Output metrics: mAP50, mAP50-95, Precision, Recall.

### Compare Models — Validation (labeled dataset)

Runs multiple weights against the same labeled dataset and prints a side-by-side accuracy table. Outputs CSV to `runs/val/comparison.csv`.

```bash
python scripts/validate/compare_val.py --config configs/compare_ppe_stecon.yaml

# Save CSV to a custom path
python scripts/validate/compare_val.py --config configs/compare_ppe_stecon.yaml --output results/my_comparison.csv
```

### Compare Models — Detection (image folder, no labels needed)

Runs multiple weights on every image in a folder and summarizes detections per class (count, avg confidence, images detected in). Saves annotated images per model to `runs/detect/<model-name>/`.

```bash
python scripts/detect/compare_detect.py --config configs/compare_detect_group.yaml
```

### Detection

```bash
# Image — using config
python scripts/detect/image.py data/images/safety1.jpg --config configs/detect_ppe_stecon.yaml

# Image — direct model path (no config needed)
python scripts/detect/image.py data/images/safety1.jpg --model runs/train/ppe-stecon/weights/best.pt

# Lower confidence threshold
python scripts/detect/image.py data/images/safety1.jpg --config configs/detect_ppe_stecon.yaml --conf 0.25

# Video
python scripts/detect/video.py data/videos/your_video.mp4 --config configs/detect_ppe_stecon.yaml

# Webcam (press Q to quit)
python scripts/detect/webcam.py --config configs/detect_ppe_stecon.yaml
```

Available overrides: `--model` `--conf`

---

## Script vs Config Quick Reference

| Goal | Script | Config |
|------|--------|--------|
| Train a model | `scripts/train/model.py` | `configs/train_ppe_stecon.yaml` |
| Validate single model (mAP) | `scripts/validate/model.py` | `configs/val_ppe_stecon.yaml` |
| Compare models on labeled dataset | `scripts/validate/compare_val.py` | `configs/compare_ppe_stecon.yaml` |
| Compare models on image folder | `scripts/detect/compare_detect.py` | `configs/compare_detect_group2.yaml` |
| Detect on single image | `scripts/detect/image.py` | `configs/detect_ppe_stecon.yaml` |
| Detect on video | `scripts/detect/video.py` | `configs/detect_ppe_stecon.yaml` |
| Live webcam detection | `scripts/detect/webcam.py` | `configs/detect_ppe_stecon.yaml` |

---

## Adding a New Experiment

1. Copy the relevant config and edit it:

   ```bash
   cp configs/train_ppe_stecon.yaml configs/train_my_experiment.yaml
   # update model, data, name, and hyperparams
   ```

2. Run:

   ```bash
   python scripts/train/model.py --config configs/train_my_experiment.yaml
   python scripts/validate/model.py --config configs/val_my_experiment.yaml
   ```

No script files need to be edited — only the config.
