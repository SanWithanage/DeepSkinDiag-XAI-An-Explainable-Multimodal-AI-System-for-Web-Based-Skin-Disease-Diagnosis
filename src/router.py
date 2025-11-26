# src/router.py
import os
import json
import argparse
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

import timm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- Safety helpers (already provided in src/utils/safety.py)
from utils.safety import (
    _validate_image_or_exit,
    _annotate_tiers_in_place,
    _maybe_add_risky_advisory,
)

# -----------------------
# Utility functions
# -----------------------

def select_device(name: str) -> torch.device:
    name = (name or "cpu").lower()
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_labels(labels_txt: Optional[str]) -> List[str]:
    labels = []
    if labels_txt and os.path.exists(labels_txt):
        with open(labels_txt, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    labels.append(line)
    return labels

def load_temperature(temp_json: Optional[str]) -> float:
    if temp_json and os.path.exists(temp_json):
        try:
            obj = json.load(open(temp_json, "r", encoding="utf-8"))
            T = float(obj.get("T", 1.0))
            if T <= 0:
                return 1.0
            return T
        except Exception:
            return 1.0
    return 1.0

def build_image_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

def load_state_dict_flexible(model: torch.nn.Module, ckpt_path: str) -> None:
    """Load state dict handling Lightning and common prefixes."""
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    new_sd = {}
    for k, v in sd.items():
        nk = k
        for pref in ("model.", "module.", "net.", "backbone."):
            if nk.startswith(pref):
                nk = nk[len(pref):]
        new_sd[nk] = v
    # Do not raise on missing/unexpected keys
    model.load_state_dict(new_sd, strict=False)

def infer_binary(
    image: Image.Image,
    ckpt: str,
    arch: str,
    device: torch.device,
    image_size: int = 224
) -> Tuple[float, float]:
    """Return (p_healthy, p_unhealthy)."""
    model = timm.create_model(arch, num_classes=2, pretrained=False)
    load_state_dict_flexible(model, ckpt)
    model.eval().to(device)

    tfm = build_image_transform(image_size)
    with torch.no_grad():
        x = tfm(image.convert("RGB")).unsqueeze(0).to(device)
        logits = model(x)
        probs = F.softmax(logits.float(), dim=-1).squeeze(0).detach().cpu().numpy()
        # Assume index 0=Healthy, 1=Unhealthy. Swap if your training differed.
        return float(probs[0]), float(probs[1])

def infer_multiclass(
    image: Image.Image,
    ckpt: str,
    arch: str,
    num_classes: int,
    device: torch.device,
    temp_T: float = 1.0,
    image_size: int = 224
) -> List[float]:
    model = timm.create_model(arch, num_classes=num_classes, pretrained=False)
    load_state_dict_flexible(model, ckpt)
    model.eval().to(device)

    tfm = build_image_transform(image_size)
    with torch.no_grad():
        x = tfm(image.convert("RGB")).unsqueeze(0).to(device)
        logits = model(x).float()
        if temp_T and temp_T > 0:
            logits = logits / temp_T
        probs = F.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy().tolist()
        return [float(p) for p in probs]

def infer_text_probs(
    text: str,
    text_ckpt: str,
    num_classes: int,
    device: torch.device
) -> Optional[List[float]]:
    """
    Returns list of class probabilities (length=num_classes), or None on failure.
    Expects a HF directory checkpoint (folder) with a classifier head for num_classes.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(text_ckpt)
        model = AutoModelForSequenceClassification.from_pretrained(text_ckpt)
        # If heads mismatch, try to continue (HF may re-init with correct num_labels if present)
        if getattr(model.config, "num_labels", None) not in (None, num_classes):
            # incompatible head; bail out gracefully
            return None
        model.eval().to(device)

        enc = tokenizer(
            text, max_length=256, truncation=True, padding=True, return_tensors="pt"
        )
        with torch.no_grad():
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits.float()
            probs = F.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy().tolist()
            if len(probs) != num_classes:
                return None
            return [float(p) for p in probs]
    except Exception:
        return None

def topk_from_probs(
    probs: List[float],
    labels: List[str],
    k: int = 5
) -> List[dict]:
    import numpy as np
    p = np.array(probs, dtype="float64")
    k = min(k, len(p))
    idx = p.argsort()[::-1][:k]
    out = []
    for i in idx:
        name = labels[i] if i < len(labels) else f"class_{i}"
        out.append({"label": name, "confidence": float(p[i])})
    return out

def fuse_probs(p_img: List[float], p_txt: Optional[List[float]], alpha: float) -> List[float]:
    import numpy as np
    pi = np.array(p_img, dtype="float64")
    if p_txt is None:
        return pi.tolist()
    pt = np.array(p_txt, dtype="float64")
    if pt.shape != pi.shape:
        return pi.tolist()
    a = max(0.0, min(1.0, float(alpha)))
    pf = (1.0 - a) * pi + a * pt
    s = pf.sum()
    if s > 0:
        pf = pf / s
    return pf.tolist()

# -------- Dynamic-alpha helpers --------

def _normalized_entropy(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    p: probabilities [B, C] or [C]; returns normalized entropy in [0,1]
    """
    if p.dim() == 1:
        p = p.unsqueeze(0)
    p = torch.clamp(p, eps, 1.0)
    H = -(p * torch.log(p)).sum(dim=-1)              # entropy
    Hmax = torch.log(torch.tensor(p.size(-1), device=p.device, dtype=p.dtype))
    return (H / Hmax).clamp(0, 1)

def _to_probs(x: torch.Tensor) -> torch.Tensor:
    """
    Ensure x is probabilities. If sums≈1, pass through; otherwise softmax.
    Accepts [C] or [B,C].
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
    row_sums = x.sum(dim=-1)
    if torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3):
        return x
    return torch.softmax(x, dim=-1)

# -----------------------
# Main
# -----------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Router: binary -> multiclass (+optional text fusion) with safety handling.")
    ap.add_argument("--image", required=True, type=str, help="Path to input image.")
    ap.add_argument("--device", type=str, default="cpu", help="cpu|cuda|mps")

    # Binary stage
    ap.add_argument("--binary_ckpt", type=str, required=False, help="Path to binary model checkpoint (.ckpt/.pt).")
    ap.add_argument("--binary_arch", type=str, default="tf_efficientnet_b0", help="timm model name for binary (num_classes=2).")
    ap.add_argument("--binary_healthy_threshold", type=float, default=0.95, help="If P(Healthy) >= threshold, skip multiclass.")

    # Multiclass stage
    ap.add_argument("--multiclass_ckpt", type=str, required=False, help="Path to multiclass model checkpoint.")
    ap.add_argument("--mc_arch", type=str, default="efficientnet_b0", help="timm model name for multiclass.")
    ap.add_argument("--labels_txt", type=str, required=False, help="Text file containing class labels (one per line).")
    ap.add_argument("--multiclass_temp", type=str, required=False, help="JSON with {'T': float} for temperature scaling.")
    ap.add_argument("--topk", type=int, default=5, help="Top-k to report.")

    # Text fusion
    ap.add_argument("--symptom_text", type=str, default="", help="Optional symptom text.")
    ap.add_argument("--text_ckpt", type=str, required=False, help="HF directory for text classifier (num_labels must match).")
    ap.add_argument("--alpha", type=float, default=0.5, help="Fusion weight for text. 0=image-only, 1=text-only.")
    ap.add_argument("--fusion_mode", type=str, default="dynamic", choices=["fixed", "dynamic"],
                    help="Use 'dynamic' entropy-based alpha (default) or 'fixed' alpha value.")

    # Output / misc
    ap.add_argument("--out", type=str, required=False, help="Where to write JSON output (optional).")
    ap.add_argument("--save_explanations", action="store_true", help="(accepted, no-op here; explanations handled elsewhere).")
    return ap

def main():
    parser = build_parser()
    args = parser.parse_args()

    # ---- SAFETY: image must be present & valid (friendly exit on failure)
    _validate_image_or_exit(args.image)

    # ---- SAFETY: Empty symptom text -> image-only (no crash)
    symptom_text = (getattr(args, "symptom_text", None) or "").strip()
    use_text = bool(symptom_text)

    device = select_device(args.device)
    labels = load_labels(args.labels_txt)
    num_classes = len(labels) if labels else 26  # fallback to 26 if labels missing
    temp_T = load_temperature(args.multiclass_temp)

    # Load image once
    image = Image.open(args.image).convert("RGB")

    # -----------------------
    # Binary stage
    # -----------------------
    binary_result = None
    stage = "binary_then_multiclass"
    p_healthy = p_unhealthy = None

    if args.binary_ckpt and os.path.exists(args.binary_ckpt):
        try:
            p_healthy, p_unhealthy = infer_binary(
                image=image,
                ckpt=args.binary_ckpt,
                arch=args.binary_arch,
                device=device,
            )
            binary_result = {
                "Healthy": round(float(p_healthy), 4),
                "Unhealthy": round(float(p_unhealthy), 4),
                "threshold_used": float(args.binary_healthy_threshold),
            }
            # If confident Healthy, skip multiclass
            if p_healthy is not None and p_healthy >= float(args.binary_healthy_threshold):
                stage = "binary_healthy"
        except Exception as e:
            # Don't crash if binary stage fails; proceed to multiclass
            binary_result = {"error": f"binary stage failed: {e}"}
    else:
        binary_result = {"note": "binary_ckpt not provided or missing — skipping binary gate."}

    # -----------------------
    # Multiclass stage
    # -----------------------
    multiclass = {}
    img_probs: Optional[List[float]] = None
    text_probs: Optional[List[float]] = None
    fused_probs: Optional[List[float]] = None

    # Run multiclass only if not short-circuited by binary healthy
    if stage != "binary_healthy" and args.multiclass_ckpt and os.path.exists(args.multiclass_ckpt):
        try:
            img_probs = infer_multiclass(
                image=image,
                ckpt=args.multiclass_ckpt,
                arch=args.mc_arch,
                num_classes=num_classes,
                device=device,
                temp_T=temp_T,
            )
            multiclass["topk_image"] = topk_from_probs(img_probs, labels, k=args.topk)
        except Exception as e:
            multiclass["error_image"] = f"multiclass image path failed: {e}"
    else:
        if stage != "binary_healthy":
            multiclass["note"] = "multiclass_ckpt not provided or missing — skipping."

    # -----------------------
    # Optional text path (guarded)
    # -----------------------
    notes: List[str] = []

    if stage != "binary_healthy" and use_text:
        if args.text_ckpt and os.path.exists(args.text_ckpt):
            tp = infer_text_probs(symptom_text, args.text_ckpt, num_classes=num_classes, device=device)
            if tp is None:
                notes.append("Text model incompatible or failed — continuing without text.")
            else:
                text_probs = tp
                multiclass["topk_text"] = topk_from_probs(text_probs, labels, k=args.topk)
        else:
            notes.append("No valid --text_ckpt provided — continuing without text.")
    elif stage != "binary_healthy" and not use_text:
        notes.append("No symptom text provided — proceeding with image-only.")

    # -----------------------
    # Fusion (if we have image probs)
    # -----------------------
    if stage != "binary_healthy" and img_probs is not None:
        if use_text and text_probs is not None and args.fusion_mode == "dynamic":
            # --- dynamic alpha path ---
            img_t = torch.tensor(img_probs, dtype=torch.float32)
            txt_t = torch.tensor(text_probs, dtype=torch.float32)

            img_p = _to_probs(img_t)  # [1,C]
            txt_p = _to_probs(txt_t)  # [1,C]

            H_img = float(_normalized_entropy(img_p).item())
            H_txt = float(_normalized_entropy(txt_p).item())

            dyn_alpha = H_img / (H_img + H_txt + 1e-8)  # trust lower-entropy modality more

            fused_t = dyn_alpha * txt_p + (1.0 - dyn_alpha) * img_p
            fused_t = fused_t.squeeze(0)
            fused_probs = fused_t.detach().cpu().numpy().tolist()

            notes.append(f"Dynamic fusion used: alpha={dyn_alpha:.3f} (H_img={H_img:.3f}, H_txt={H_txt:.3f})")
        else:
            # --- fallback to fixed alpha (image-only or text missing or forced fixed mode) ---
            fused_probs = fuse_probs(img_probs, text_probs if use_text else None, float(args.alpha))
            if args.fusion_mode == "fixed" and use_text and text_probs is not None:
                notes.append(f"Fixed fusion used: alpha={float(args.alpha):.3f}")

        if fused_probs is not None:
            multiclass["topk_fused"] = topk_from_probs(fused_probs, labels, k=args.topk)

    # -----------------------
    # Compose result
    # -----------------------
    res = {
        "image": args.image,
        "stage": ("fusion" if (stage != "binary_healthy" and (fused_probs is not None)) else stage),
        "binary": binary_result,
        "multiclass": multiclass,
    }
    if notes:
        res["notes"] = notes

    # ---- SAFETY: add confidence tiers + risky label advisory
    _annotate_tiers_in_place(res)
    _maybe_add_risky_advisory(res, getattr(args, "labels_txt", None))

    # Write/print JSON
    out_json = json.dumps(res, indent=2)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out_json)
    print(out_json)

if __name__ == "__main__":
    main()
