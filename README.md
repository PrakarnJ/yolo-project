# PPE Detection — Iterative Retraining Pipeline

YOLO11 + RF-DETR PPE (Personal Protective Equipment) detection for industrial safety monitoring (EGAT/Stecon).  
The pipeline is designed for continuous improvement: drop a new labeled batch, run one command, and always know which model is best — across both frameworks.

**11 detection classes:** `helmet` · `longsleeve` · `shortsleeve` · `coverall` · `longpant` · `shortpant` · `skirt` · `vest` · `glove` · `boot` · `shoe`

---

## Project Structure

```
yolo-project/
├── configs/
│   └── retrain/
│       └── ppe.yaml              # training, validation, and detection settings
├── scripts/
│   └── retrain.py                # main pipeline script
├── datasets/                     # INPUT: drop labeled batch folders here (git-ignored)
│   └── ppe_stecon/
│       └── egat_uat/             # first batch — 800 train / 200 val
├── workspace/                    # AUTO-GENERATED outputs (git-ignored)
│   ├── merged/                   # YOLO dataset built from all batches (symlinks)
│   ├── merged_coco/              # COCO format conversion for RF-DETR
│   ├── runs/                     # one folder per training run
│   ├── leaderboard.csv           # accuracy history — YOLO and RF-DETR ranked together
│   └── detection_history.csv     # detection stats per run
├── data/
│   └── images/                   # unlabeled test images for detection comparison
│       ├── group-1/
│       ├── group-2/
│       └── group-3/              # ← used by default in ppe.yaml
├── models/
│   └── raw_weight/
│       └── yolo11n.pt            # base YOLO11n weights (not committed — download separately)
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

Requirements: Python 3.8+, CUDA-capable GPU.

Download base weights if not present:

```bash
# yolo11n.pt is auto-downloaded by ultralytics on first use,
# or place it manually at models/raw_weight/yolo11n.pt
```

---

## Workflow

### 1 — Prepare a dataset batch

Each batch is a folder inside `datasets/` with this structure:

```
datasets/
└── my_batch/
    ├── train/
    │   ├── images/
    │   └── labels/       ← YOLO .txt format
    ├── val/
    │   ├── images/
    │   └── labels/
    └── dataset.yaml      ← must list class names in the correct order
```

**`dataset.yaml` minimum content:**

```yaml
names:
  - helmet
  - longsleeve
  - shortsleeve
  - coverall
  - longpant
  - shortpant
  - skirt
  - vest
  - glove
  - boot
  - shoe
```

> Class names and order must be identical across every batch.

### 2 — Run the pipeline

```bash
# Standard run: merge → train YOLO + RF-DETR → evaluate all models → update leaderboard
.venv/bin/python scripts/retrain.py --config configs/retrain/ppe.yaml

# Re-evaluate all existing models without training a new one
.venv/bin/python scripts/retrain.py --config configs/retrain/ppe.yaml --eval-only

# Skip RF-DETR training (YOLO only)
.venv/bin/python scripts/retrain.py --config configs/retrain/ppe.yaml --skip-rfdetr

# Override YOLO epochs without editing the config
.venv/bin/python scripts/retrain.py --config configs/retrain/ppe.yaml --epochs 100
```

### 3 — Read the summary

At the end of each run the script prints a ranked table:

```
--- VALIDATION (labeled val set) ---
Rank  Run                         Frm     Batches     mAP50   mAP50-95   Prec    Recall
 1    ★ run_002_... [NEW]         rfdetr  egat_uat,…  0.941*   0.661*     N/A      N/A
 2      run_002_... [NEW]         yolo    egat_uat,…  0.879    0.637    0.912*   0.801*
 3      run_001_...               rfdetr  egat_uat    0.936    0.655     N/A      N/A
 4      run_001_...               yolo    egat_uat    0.857    0.604    0.916    0.778

--- DETECTION (image folder: data/images/group-3) ---
Rank  Run                         Frm     Total Det  Imgs w/ Det  Top Class
 1    ★ run_002_... [NEW]         rfdetr       342*       58/64   helmet(120)
 2      run_002_... [NEW]         yolo          45        18/32   helmet(20)
 3      run_001_...               rfdetr        28        16/32   helmet(13)
 4      run_001_...               yolo           2         2/32   boot(2)

Best model: RFDETR  workspace/runs/run_002_.../rfdetr/checkpoint_best_total.pth
```

`★` = overall best · `*` = best value in column · `Frm` = framework

---

## Config Reference

**`configs/retrain/ppe.yaml`**

```yaml
classes:              # must match dataset.yaml in every batch
  - helmet
  - ...

datasets_dir: datasets/ppe_stecon   # where you drop batch folders
workspace_dir: workspace            # all generated outputs land here

rfdetr_train:
  enabled: true        # set false (or --skip-rfdetr) to disable
  model: base          # base | large | small | medium | nano
  epochs: 50
  batch_size: 4
  grad_accum_steps: 4
  lr: 0.0001
  resolution: 560

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

detect:
  images: data/images/group-3   # same folder every run — enables fair visual comparison
  conf: 0.5
  imgsz: 720
```

---

## Outputs

```
workspace/
├── merged/
│   ├── train/images/    ← symlinks: <batch>__<filename>
│   ├── train/labels/
│   ├── val/images/
│   ├── val/labels/
│   └── merged.yaml
├── merged_coco/         ← COCO format for RF-DETR (auto-generated from merged/)
│   ├── train/  (_annotations.coco.json + image symlinks)
│   └── valid/  (_annotations.coco.json + image symlinks)
├── runs/
│   └── run_001_20260611_180000/
│       ├── weights/
│       │   ├── best.pt                    ← YOLO weights
│       │   └── last.pt
│       ├── rfdetr/
│       │   └── checkpoint_best_total.pth  ← RF-DETR weights
│       ├── run_meta.yaml                  ← batches used, image counts, timestamp
│       └── detect/                        ← annotated images from the test folder
├── leaderboard.csv           ← one row per (run, framework), ranked by mAP50
└── detection_history.csv     ← detection totals per run
```

---

## Current Datasets

| Batch | Train | Val | Source |
|-------|------:|----:|--------|
| `egat_uat` | 800 | 200 | Sampled from EGAT UAT crop dataset (2025-01-13) |

---

## Adding a New Batch

1. Prepare the folder with the structure above
2. Drop it into `datasets/`
3. Run the script — it automatically merges everything and trains from scratch on the full combined dataset:

```bash
.venv/bin/python scripts/retrain.py --config configs/retrain/ppe.yaml
```

The new model is validated against all previous models on the same merged val set so the comparison is always fair.
