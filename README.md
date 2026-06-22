# Object Detection Pipeline — YOLO11 & RF-DETR

Iterative retraining and interactive inference for two detection domains:

| Domain | Classes | Config |
|--------|---------|--------|
| **PPE** (Personal Protective Equipment) | `helmet` · `longsleeve` · `shortsleeve` · `coverall` · `longpant` · `shortpant` · `skirt` · `vest` · `glove` · `boot` · `shoe` | `configs/retrain/ppe.yaml` |
| **Physical Fight / Fall** | `fight` · `fall` | `configs/retrain/physical_fight.yaml` |

---

## Project Structure

```
yolo-project/
├── configs/retrain/
│   ├── ppe.yaml                  # PPE training & validation settings
│   └── physical_fight.yaml       # Fight/fall training & validation settings
├── scripts/
│   ├── retrain.py                # Iterative training pipeline
│   ├── blind_test.py             # Evaluate models on an external image folder
│   └── app.py                    # Gradio inference UI
├── models/raw_weight/
│   └── yolo11n.pt                # COCO-pretrained base weights (person detector)
├── datasets/                     # INPUT: drop labeled batch folders here (git-ignored)
│   ├── ppe_stecon/
│   └── physical_fight/
├── workspace_ppe/                # AUTO-GENERATED PPE outputs (git-ignored)
│   ├── merged/                   # Merged YOLO dataset (symlinks)
│   ├── merged_coco/              # COCO format for RF-DETR
│   ├── runs/                     # One folder per training run
│   ├── leaderboard.csv           # Validation rankings across all runs
│   └── blind_test_leaderboard.csv
├── workspace_fight/              # AUTO-GENERATED fight/fall outputs (git-ignored)
│   └── ...                       # Same structure as workspace_ppe/
├── requirements.txt
└── .gitignore
```

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requirements: Python 3.10+, CUDA-capable GPU.

---

## Inference UI

An interactive web UI for running trained models on images, videos, and live camera.

```bash
.venv/bin/python scripts/app.py
# Open http://localhost:7860
```

### Features

**Framework filter** — Radio buttons (`All` / `YOLO` / `RF-DETR`) instantly filter the model dropdown to show only models of the selected type.

**Model selector (multi-select)** — Auto-discovers all trained models from `workspace_ppe/` and `workspace_fight/`. Select one model for standard inference or multiple models to compare them side-by-side. Dropdown entries follow the format:
```
<domain> | <run_name> | <framework>
# e.g.  ppe | run_001_20260615_212600 | YOLO
#        fight | run_001_20260616_210059 | RF-DETR
```

**Input modes** (tabs):
| Tab | Single model | Multiple models |
|-----|-------------|-----------------|
| Image | Annotated output image + stats table | All model outputs shown simultaneously in separate columns, each with its own annotated image and detection table |
| Video | Downloadable annotated `.mp4` + stats table | One video player per model shown side-by-side, each with its own stats table |
| Camera | Live annotated stream | All models stitched side-by-side in real-time with label bars |

**Person-crop pipeline** — When *Crop by person first* is checked, each frame is processed in two stages:
1. A COCO-pretrained `yolo11n.pt` detects all persons in the scene (green boxes).
2. Each person crop is passed individually to the selected task model(s).

This improves accuracy when the task model was trained on person crops rather than full scenes. Automatically falls back to full-frame inference when no persons are detected.

**Detection summary** — After inference, a table is shown below each model's output:

| Class | Count | Max Conf | Avg Conf |
|-------|-------|----------|----------|
| helmet | 3 | 0.91 | 0.85 |
| shortsleeve | 2 | 0.78 | 0.72 |

When comparing multiple models, each column has its own independent table so results can be read at a glance. For video, the table aggregates detections across all frames.

**Controls:**

| Control | Default | Description |
|---------|---------|-------------|
| Framework | All | Filter model dropdown by YOLO or RF-DETR |
| Model(s) | first discovered | Select one or more trained models; selecting multiple enables side-by-side comparison |
| Confidence | 0.5 | Detection confidence threshold |
| Crop by person first | ✓ | Enable two-stage person-crop pipeline |
| Person Conf | 0.3 | Confidence for the person detector stage |

---

## Training Pipeline

### 1 — Prepare a dataset batch

Each batch is a folder placed inside the appropriate `datasets/` subdirectory:

```
datasets/ppe_stecon/my_new_batch/
├── train/
│   ├── images/
│   └── labels/       ← YOLO .txt format (class cx cy w h, normalised)
├── val/
│   ├── images/
│   └── labels/
└── dataset.yaml      ← must list class names in the exact order defined in the config
```

> Class names and order must match `classes:` in `configs/retrain/ppe.yaml` exactly.

### 2 — Run the pipeline

```bash
# Merge datasets → train YOLO11 + RF-DETR → evaluate all models → update leaderboard
.venv/bin/python scripts/retrain.py --config configs/retrain/ppe.yaml

# Fight/fall domain
.venv/bin/python scripts/retrain.py --config configs/retrain/physical_fight.yaml

# Re-evaluate all existing models without training a new one
.venv/bin/python scripts/retrain.py --config configs/retrain/ppe.yaml --eval-only

# Skip RF-DETR (YOLO only)
.venv/bin/python scripts/retrain.py --config configs/retrain/ppe.yaml --skip-rfdetr

# Override YOLO epochs
.venv/bin/python scripts/retrain.py --config configs/retrain/ppe.yaml --epochs 100
```

### 3 — Blind test on external images

```bash
# Run all trained models on an external folder; optionally provide labels for mAP
.venv/bin/python scripts/blind_test.py \
    --config configs/retrain/ppe.yaml \
    --test-images path/to/images/ \
    [--labels path/to/labels/]
```

Results are saved to `workspace_ppe/external_blind_test_<timestamp>/`.

### 4 — Read the leaderboard

The pipeline prints a ranked table at the end of each run and writes it to `leaderboard.csv`:

```
Rank  Run                      Frm     mAP50   mAP50-95   Prec    Recall
 1  ★ run_002_... [NEW]        rfdetr  0.842*   0.612*     N/A      N/A
 2    run_002_... [NEW]        yolo    0.779    0.581    0.901*   0.812*
 3    run_001_...              rfdetr  0.831    0.601     N/A      N/A
 4    run_001_...              yolo    0.764    0.558    0.888    0.795
```

`★` = current best · `*` = best value in column

---

## Config Reference

```yaml
# configs/retrain/ppe.yaml
classes:
  - helmet          # class 0
  - longsleeve      # class 1
  - ...             # must match dataset.yaml in every batch

datasets_dir: datasets/ppe_stecon
workspace_dir: workspace_ppe

rfdetr_train:
  enabled: true
  model: nano           # nano | small | base | large
  epochs: 50
  batch_size: 4
  grad_accum_steps: 4
  lr: 0.0001
  resolution: 576

train:
  base_model: models/raw_weight/yolo11n.pt
  epochs: 300
  imgsz: 720
  batch: 16
  freeze: 10
  lr0: 0.001
  patience: 20

validate:
  conf: 0.5
  iou: 0.5
  imgsz: 720
  batch: 16
```

---

## Run Output Structure

```
workspace_ppe/runs/run_001_20260615_212600/
├── weights/
│   ├── best.pt                       ← YOLO11 best checkpoint
│   └── last.pt
├── rfdetr/
│   └── checkpoint_best_total.pth     ← RF-DETR best checkpoint
├── run_meta.yaml                     ← batches used, image counts, timestamp
├── detect_test_yolo/                 ← annotated test images (YOLO)
└── detect_test_rfdetr/               ← annotated test images (RF-DETR)
```

---

## Adding a New Batch

1. Prepare the batch folder (see structure above).
2. Drop it into `datasets/ppe_stecon/` (or `datasets/physical_fight/`).
3. Run the pipeline — it merges everything, trains from scratch on the full combined dataset, and re-evaluates all past models on the same val set so rankings stay fair.

```bash
.venv/bin/python scripts/retrain.py --config configs/retrain/ppe.yaml
```
