# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Object detection pipeline (YOLO11 + RF-DETR) for two domains, each driven by a config in `configs/retrain/`:

- **PPE detection** (`configs/retrain/ppe.yaml`) — 11 classes in this exact order: `helmet, longsleeve, shortsleeve, coverall, longpant, shortpant, skirt, vest, glove, boot, shoe`. Class order matters: labels are index-based, so never reorder this list.
- **Fight/fall detection** (`configs/retrain/physical_fight.yaml`) — 2 classes: `fight, fall`.

## Environment

Always use the project venv interpreter directly — never system `python3`:

```bash
.venv/bin/python <script>
```

Setup (Python 3.10+, CUDA GPU expected):

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Base weights in `models/raw_weight/` (`yolo11n.pt`, pose models) are git-ignored and downloaded separately.

## Commands

```bash
# Full retraining pipeline (YOLO + RF-DETR, eval, leaderboard, blind test)
.venv/bin/python scripts/retrain.py --config configs/retrain/ppe.yaml
# Flags: --eval-only (skip training), --skip-rfdetr, --epochs N

# Blind test all trained models on an external image folder
.venv/bin/python scripts/blind_test.py --config configs/retrain/ppe.yaml \
    --test-images path/to/images [--labels path/to/labels]

# Gradio debugging workbench (image/batch/video/camera, up to 4 models side by side)
.venv/bin/python scripts/app.py   # http://localhost:7860
```

There is no test suite, linter, or build system — this is a scripts-and-configs research project.

## Architecture

Three active scripts in `scripts/`; everything else flows from the two YAML configs.

**`scripts/retrain.py`** — single-file, 8-phase pipeline:
1. Scan `datasets/<domain>/` batch folders, verify class consistency across `dataset.yaml`s
2. Merge all batches via symlinks into `<workspace>/merged/` with a 90/10 val/test split (recorded in `test_split_manifest.json`); also converts to COCO in `<workspace>/merged_coco/` for RF-DETR
3. Train YOLO into `<workspace>/runs/run_NNN_<timestamp>/`, then RF-DETR into `<run_dir>/rfdetr/`
4. Re-evaluate ALL past runs on the current val set (keeps rankings comparable)
5. Update `<workspace>/leaderboard.csv`
6. Blind test every model on the held-out test split → `blind_test_leaderboard.csv`
7. Blind test every model on each subfolder of `datasets/<domain>/custom_blind_test/` (auto-created with a `default/` starter subfolder, git-ignored). Any subfolder with an `images/` dir is treated as its own named test set — add as many as you want (e.g. `custom_blind_test/site_a/images/`, `custom_blind_test/site_b/images/`); `labels/` per subfolder is optional and enables YOLO mAP. Results for all test sets land in one `custom_blind_test_leaderboard.csv` (keyed by a `test_set` column) and print as separate ranked tables; a subfolder with no images is skipped
8. Print ranked summary

**`scripts/app.py`** — thin entry point for the Gradio debugging workbench, implemented in `scripts/workbench/`:
- `registry.py` — `discover_models()` auto-scans `workspace_ppe/` and `workspace_fight/` for YOLO `best.pt` and RF-DETR `.pth` checkpoints, plus `*pose*.pt` in `models/raw_weight/` (`WORKSPACE_CONFIGS` maps workspace→config). Also: arbitrary weight loading (`register_custom_weight`, class names auto-read from YOLO checkpoints), run metadata (`get_run_info`), `ModelCache` (one GPU model at a time), `PersonDetector`.
- `inference.py` — two paths: `predict_raw`/`filter_preds`/`render_preds` (Image & Batch tabs: infer once at `BASE_CONF=0.05`, cache raw arrays, re-filter at any threshold with zero re-inference) and `dispatch` (Video/Camera: conf is the inference threshold). Crop mode offsets per-crop boxes back to full-image coords.
- `matching.py` — IoU matrix, greedy GT matching (TP/FP/FN + per-image P/R), cross-model disagreement clustering.
- `ui_image.py` / `ui_batch.py` / `ui_video.py` / `ui_camera.py` / `ui_shared.py` / `ui.py` — tabs: threshold explorer with GT overlay + disagreement view + crop-debug gallery (Image), folder gallery with cached re-ranking (Batch), detection timeline + frame scrubber reading annotated mp4s (Video), live streaming (Camera).

**Data flow / directories:**
- `datasets/<domain>/<batch-name>/` — INPUT: drop labeled YOLO batches here, each with `train/{images,labels}`, `val/{images,labels}`, `dataset.yaml`. The pipeline picks up all batches automatically on the next run.
- `workspace_ppe/`, `workspace_fight/` — AUTO-GENERATED per-domain outputs (merged datasets, runs, leaderboards). Don't hand-edit.
- `runs/mlflow/` — MLflow tracking output from Ultralytics training.
- `old_pipeline/` — legacy pre-consolidation pipeline (git-ignored). Do not extend it; `scripts/retrain.py` is its successor.

## Gotchas

- The retrain pipeline always trains from scratch on the full merged dataset each run — there is no incremental fine-tuning of previous runs.
- `--labels` in `blind_test.py` enables mAP computation for YOLO models only.
- A stray `yolo11n.pt` at the repo root is auto-downloaded by Ultralytics; the canonical copy is `models/raw_weight/yolo11n.pt`.
