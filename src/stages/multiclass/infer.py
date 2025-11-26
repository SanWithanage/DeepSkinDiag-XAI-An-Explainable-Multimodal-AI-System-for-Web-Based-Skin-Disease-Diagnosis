import importlib
import numpy as np

# This module exposes the two names the router expects.
# It forwards to whatever exists inside your current infer_wrap.

_CANDIDATE_FUNCS = [
    # probs-returning names first
    "predict_probs", "predict", "predict_one", "infer_probs", "infer_image",
    "infer_one", "run_one", "run_image", "probs", "infer",
]

def _call_candidate(fn, ckpt, image, device):
    # try (ckpt, image, device) then (image, ckpt, device)
    try:
        out = fn(ckpt, image, device)
    except TypeError:
        out = fn(image, ckpt, device)
    arr = np.asarray(out, dtype="float64").ravel()
    s = arr.sum()
    if s > 0: arr = arr / s
    return arr

def predict_probs(ckpt, image, device):
    m = importlib.import_module("src.stages.multiclass.infer_wrap")
    for name in _CANDIDATE_FUNCS:
        fn = getattr(m, name, None)
        if callable(fn):
            return _call_candidate(fn, ckpt, image, device)
    raise RuntimeError("infer_wrap has no usable predict function (tried: %s)" % ", ".join(_CANDIDATE_FUNCS))

def predict_logits(ckpt, image, device):
    # If your infer_wrap already provides logits, prefer that name:
    m = importlib.import_module("src.stages.multiclass.infer_wrap")
    for name in ["predict_logits", "logits", "infer_logits"]:
        fn = getattr(m, name, None)
        if callable(fn):
            try:
                return _call_candidate(fn, ckpt, image, device)
            except Exception:
                pass
    # Fallback: get probs and convert to log-space pseudo-logits
    p = predict_probs(ckpt, image, device)
    eps = 1e-12
    return np.log(np.clip(p, eps, 1.0))
