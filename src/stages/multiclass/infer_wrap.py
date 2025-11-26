import os, torch, torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

# ---- SET THESE to match your training ----
NUM_CLASSES = 26
MODEL_NAME = "tf_efficientnet_b0"  # change if you trained b2/b3/etc.

# Try to reuse your eval/build code if present; else fall back to timm
def _build_model(num_classes=NUM_CLASSES, model_name=MODEL_NAME):
    try:
        # If your eval script exposes a builder, prefer it:
        # e.g., from scripts.eval_multiclass import build_model
        from scripts.eval_multiclass import build_model  # adjust if name differs
        return build_model(num_classes=num_classes, model_name=model_name)
    except Exception:
        import timm
        return timm.create_model(model_name, pretrained=False, num_classes=num_classes)

_TF = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def _load_image(p):
    im = Image.open(p).convert("RGB")
    x = _TF(im).unsqueeze(0)
    return x

def _to_device(x, device: str):
    dev = device
    if dev == "cuda" and not torch.cuda.is_available():
        dev = "cpu"
    if dev == "mps":
        if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
            dev = "cpu"
    return x.to(dev), dev

def _restore(model, ckpt):
    sd = torch.load(ckpt, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    # Try exact load; if keys are prefixed (e.g., "model."), strip first segment on mismatch
    try:
        model.load_state_dict(sd, strict=True)
    except Exception:
        new_sd = {}
        for k,v in sd.items():
            new_sd[k.split(".",1)[-1]] = v
        model.load_state_dict(new_sd, strict=False)
    return model

def predict_logits(ckpt: str, image_path: str, device: str):
    if not os.path.exists(image_path):
        raise RuntimeError(f"Image not found: {image_path}")
    if not os.path.exists(ckpt):
        raise RuntimeError(f"Checkpoint not found: {ckpt}")

    model = _build_model(NUM_CLASSES, MODEL_NAME)
    model = _restore(model, ckpt)
    model.eval()

    x = _load_image(image_path)
    x, dev = _to_device(x, device)
    model.to(dev)

    with torch.no_grad():
        y = model(x)              # (1, C)
        y = y.squeeze(0)          # (C,)
        logits = y.detach().cpu().numpy().astype("float64")
    return logits

def predict_probs(ckpt: str, image_path: str, device: str):
    logits = predict_logits(ckpt, image_path, device)
    x = torch.from_numpy(logits)
    p = torch.softmax(x, dim=-1).cpu().numpy().astype("float64")
    s = p.sum()
    return p / s if s > 0 else p
