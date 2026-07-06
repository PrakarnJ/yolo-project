"""Model discovery, registry, and caching for the debugging workbench."""
import csv
import warnings
from pathlib import Path

import numpy as np
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent.parent

WORKSPACE_CONFIGS = {
    "ppe":   (ROOT / "workspace_ppe",   ROOT / "configs/retrain/ppe.yaml"),
    "fight": (ROOT / "workspace_fight", ROOT / "configs/retrain/physical_fight.yaml"),
}


def _load_class_names(config_path: Path) -> list:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    names = []
    for item in cfg.get("classes", []):
        for part in str(item).split(","):
            part = part.strip()
            if part:
                names.append(part)
    return names


def discover_models() -> dict:
    registry = {}
    for ws_key, (ws_dir, cfg_path) in WORKSPACE_CONFIGS.items():
        runs_dir = ws_dir / "runs"
        if not runs_dir.exists():
            continue
        class_names = _load_class_names(cfg_path)
        for run_dir in sorted(d for d in runs_dir.iterdir() if d.is_dir()):
            yolo_w   = run_dir / "weights" / "best.pt"
            rfdetr_w = run_dir / "rfdetr"  / "checkpoint_best_total.pth"
            if yolo_w.exists():
                label = f"{ws_key} | {run_dir.name} | YOLO"
                registry[label] = {"framework": "yolo", "weights": str(yolo_w),
                                   "class_names": class_names,
                                   "workspace": ws_key, "run_dir": str(run_dir)}
            if rfdetr_w.exists():
                label = f"{ws_key} | {run_dir.name} | RF-DETR"
                registry[label] = {"framework": "rfdetr", "weights": str(rfdetr_w),
                                   "class_names": class_names,
                                   "workspace": ws_key, "run_dir": str(run_dir)}
    for pose_w in sorted((ROOT / "models/raw_weight").glob("*pose*.pt")):
        label = f"pose | {pose_w.stem} | YOLO-Pose"
        registry[label] = {"framework": "yolo-pose", "weights": str(pose_w),
                           "class_names": ["person"]}
    return registry


MODEL_REGISTRY = discover_models()


class ModelCache:
    """Keeps one task model loaded at a time to avoid exhausting GPU memory."""

    def __init__(self):
        self._key   = None
        self._model = None

    def get(self, key: str):
        if self._key == key:
            return self._model
        entry = MODEL_REGISTRY[key]
        self._model = None  # release old
        if entry["framework"] in ("yolo", "yolo-pose"):
            self._model = YOLO(entry["weights"])
        else:
            warnings.filterwarnings("ignore", category=FutureWarning, module="rfdetr")
            from rfdetr import RFDETR
            self._model = RFDETR.from_checkpoint(entry["weights"])
        self._key = key
        return self._model

    def seed(self, key: str, model):
        """Adopt an already-loaded model (e.g. right after a custom-weight load)."""
        self._key   = key
        self._model = model


class PersonDetector:
    """Lazy-loads the COCO-pretrained YOLO model for person detection (class 0)."""

    def __init__(self):
        self._model = None

    def _get(self):
        if self._model is None:
            self._model = YOLO(str(ROOT / "models/raw_weight/yolo11n.pt"))
        return self._model

    def detect(self, frame_bgr: np.ndarray, conf: float = 0.3) -> list:
        """Return list of [x1, y1, x2, y2] for every detected person."""
        result = self._get()(frame_bgr, classes=[0], conf=conf, verbose=False)[0]
        return [b.xyxy[0].tolist() for b in result.boxes]


model_cache     = ModelCache()
person_detector = PersonDetector()


def register_custom_weight(path_str: str, class_override: str = "") -> tuple:
    """Add an arbitrary local weight file to the registry. Returns (label, status_md)."""
    path = Path(path_str.strip()).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"weight file not found: {path}")
    override = [p.strip() for p in class_override.split(",") if p.strip()]

    if path.suffix == ".pth":
        framework, fw_tag = "rfdetr", "RF-DETR"
        names = override
        note  = "" if override else " — ⚠️ no class names given, raw class ids will be shown"
        model = None
    elif path.suffix == ".pt":
        model = YOLO(str(path))  # loads once; adopted by the cache below
        if model.task == "pose":
            framework, fw_tag = "yolo-pose", "YOLO-Pose"
        else:
            framework, fw_tag = "yolo", "YOLO"
        embedded = model.names
        embedded = list(embedded.values()) if isinstance(embedded, dict) else list(embedded)
        names = override or embedded
        note  = (" — classes from override" if override
                 else f" — {len(names)} classes read from checkpoint")
    else:
        raise ValueError(f"unsupported weight type '{path.suffix}' (expected .pt or .pth)")

    label = f"custom | {path.stem} | {fw_tag}"
    MODEL_REGISTRY[label] = {"framework": framework, "weights": str(path),
                             "class_names": names, "source": "custom"}
    if model is not None:
        model_cache.seed(label, model)
    return label, f"✅ Loaded **{label}**{note}"


_run_info_cache = {}


def get_run_info(key: str) -> str:
    """Markdown summary of a model's provenance: run_meta.yaml + leaderboard row + epochs."""
    if key in _run_info_cache:
        return _run_info_cache[key]

    entry   = MODEL_REGISTRY.get(key, {})
    run_dir = entry.get("run_dir")
    lines   = [f"**{key}**", f"- weights: `{entry.get('weights', '?')}`",
               f"- classes: {len(entry.get('class_names', []))}"]

    if not run_dir:
        src = "custom weight" if entry.get("source") == "custom" else "base weight"
        lines.append(f"- {src} — no run metadata")
        info = "\n".join(lines)
        _run_info_cache[key] = info
        return info

    meta_p = Path(run_dir) / "run_meta.yaml"
    if meta_p.exists():
        meta = yaml.safe_load(meta_p.read_text()) or {}
        lines.append(f"- batches: {', '.join(map(str, meta.get('batches', [])))}")
        lines.append(f"- images: train {meta.get('train_images')} / "
                     f"val {meta.get('val_images')} / test {meta.get('test_images')}")
        lines.append(f"- trained: {meta.get('timestamp')}")

    fw = entry["framework"]
    ws_dir, cfg_path = WORKSPACE_CONFIGS[entry["workspace"]]
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    train_block = cfg.get("rfdetr_train" if fw == "rfdetr" else "train") or {}
    lines.append(f"- configured epochs: {train_block.get('epochs', '?')}")

    lb = ws_dir / "leaderboard.csv"
    if lb.exists():
        run_name = Path(run_dir).name
        with open(lb) as f:
            for row in csv.DictReader(f):
                if row.get("run") == run_name and row.get("framework") == fw:
                    best = " | ⭐ best" if str(row.get("is_best")).lower() in ("true", "1") else ""
                    lines.append(f"- mAP50 {row.get('mAP50')} | mAP50-95 {row.get('mAP50-95')} | "
                                 f"P {row.get('Precision')} | R {row.get('Recall')}{best}")
                    break

    info = "\n".join(lines)
    _run_info_cache[key] = info
    return info
