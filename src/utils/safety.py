# src/utils/safety.py
import os, sys, re
from typing import Optional
from PIL import Image, UnidentifiedImageError

# ---- Safety/failure-handling config (edit if needed)
RISKY_LABELS_RAW = {"SkinCancer", "Melanoma", "Bullous"}  # only those present in labels_26 will be used
HIGH_THRESH = 0.70
MED_THRESH  = 0.50

def _norm(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', s.lower()).strip('_')

def _confidence_tier(p: float) -> str:
    if p >= HIGH_THRESH: return "High"
    if p >= MED_THRESH:  return "Medium"
    return "Low"

def _validate_image_or_exit(path: str):
    """Friendly error + exit(1) if image missing/corrupt"""
    if not path:
        print("[Error] No image path provided. Please pass --image <file>.", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(path) or not os.path.isfile(path):
        print(f"[Error] Image not found: {path}\nTip: check the path or quotes if it has spaces.", file=sys.stderr)
        sys.exit(1)
    try:
        with Image.open(path) as im:
            im.verify()  # integrity check
    except (UnidentifiedImageError, OSError) as e:
        print(f"[Error] The file is not a valid image or is corrupted: {path}\nDetails: {e}", file=sys.stderr)
        sys.exit(1)

def _available_labels_from_file(labels_txt: Optional[str]) -> set[str]:
    s: set[str] = set()
    try:
        if labels_txt and os.path.exists(labels_txt):
            with open(labels_txt) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        s.add(_norm(line))
    except Exception:
        pass
    return s

def _annotate_tiers_in_place(res: dict):
    """Add confidence_tier to any top-k lists present."""
    mc = res.get("multiclass") or {}
    for key in ("topk_fused", "topk_image", "topk_text"):
        if key in mc and isinstance(mc[key], list):
            for item in mc[key]:
                if "confidence" in item:
                    try:
                        item["confidence_tier"] = _confidence_tier(float(item["confidence"]))
                    except Exception:
                        pass

def _get_topk_view(res: dict):
    """Prefer fused, else image, else text."""
    mc = res.get("multiclass") or {}
    for key in ("topk_fused", "topk_image", "topk_text"):
        if key in mc and mc[key]:
            return mc[key]
    return []

def _maybe_add_risky_advisory(res: dict, labels_txt: Optional[str]):
    """Emit advisory if any risky labels appear in top-3."""
    avail = _available_labels_from_file(labels_txt)
    effective_risky = {_norm(x) for x in RISKY_LABELS_RAW}
    if avail:
        # only keep risky labels that actually exist in your 26-label set
        effective_risky = {r for r in effective_risky if r in avail}

    topk = _get_topk_view(res)[:3]
    hits = [d for d in topk if _norm(d.get("label", "")) in effective_risky]
    if hits:
        names = ", ".join(h["label"] for h in hits)
        res["advisory"] = (
            f"⚠️ Advisory: {names} appeared in the top-3 predictions. "
            "These can be serious. This tool is not a medical device—seek qualified clinical evaluation."
        )
        res["risky_hits"] = [h["label"] for h in hits]
