"""Warm-start loading that accepts every checkpoint layout the project has produced.

The DMC weights are loaded with evaluate_vcm.load_codec_checkpoint, the same
function that evaluated the paper checkpoint, so any file it could evaluate
also warm-starts training here.
"""

import json
from pathlib import Path

import torch


CLONE_KEYS = ("cloned_frontend_state_dict", "student_front")
DMC_KEYS = ("dmc_state_dict", "p_net", "model_state_dict", "state_dict")


def _is_tensor_dict(value):
    return isinstance(value, dict) and bool(value) and all(torch.is_tensor(v) for v in value.values())


def _strip_module(state):
    return {key.removeprefix("module."): value for key, value in state.items()}


def load_raw(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def clone_state(checkpoint):
    for key in CLONE_KEYS:
        if _is_tensor_dict(checkpoint.get(key)):
            return key, _strip_module(checkpoint[key])
    return None, None


def warm_start(video_model, det_clone, path):
    """Load DMC (and the detection clone when present). Returns (checkpoint, clone_source)."""
    from evaluate_vcm import load_codec_checkpoint

    checkpoint = load_codec_checkpoint(video_model, path)
    key, state = clone_state(checkpoint)
    if det_clone is not None and state is not None:
        det_clone.load_state_dict(state)
        return checkpoint, f"init_checkpoint:{key}"
    return checkpoint, "pretrained_yolov5s"


def _flatten(data, prefix=""):
    if isinstance(data, dict):
        for key, value in data.items():
            yield from _flatten(value, f"{prefix}{key}.")
    else:
        yield prefix.rstrip("."), data


def lambda_evidence(path, checkpoint):
    """Collect everything that says which lambda range a checkpoint was trained with."""
    evidence = {}
    if isinstance(checkpoint, dict):
        for key, value in checkpoint.items():
            if "lambda" in key.lower() and not torch.is_tensor(value):
                evidence[f"checkpoint:{key}"] = value
    path = Path(path)
    for folder in (path.parent, path.parent.parent):
        config = folder / "config.json"
        if not config.is_file():
            continue
        try:
            data = json.loads(config.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        for key, value in _flatten(data):
            if "lambda" in key.lower():
                evidence[f"{config.name}:{key}"] = value
    evidence["path"] = str(path)
    return evidence


def is_lambda_1_64(evidence):
    """True only when the evidence points at the paper's unshaped 1-64 schedule."""
    text = json.dumps({k: v for k, v in evidence.items() if k != "path"}, default=str).lower()
    if "shaped" in text or "0.25" in text:
        return False
    numbers = set()
    for key, value in evidence.items():
        if key == "path":
            continue
        values = value if isinstance(value, (list, tuple)) else [value]
        for item in values:
            if isinstance(item, (int, float)) and not isinstance(item, bool):
                numbers.add(float(item))
    if {1.0, 64.0} <= numbers:
        return True
    name = evidence["path"].lower().replace("-", "_")
    return not numbers and "lambda_1_64" in name
