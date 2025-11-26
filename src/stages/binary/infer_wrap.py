import os, torch, torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# ---- SET THESE to match your binary training ----
MODEL_NAME = "tf_efficientnet_b0"   # change to the variant you trained (e.g., tf_efficientnet_b2)
NUM_CLASSES = 2

# If you have a custom builder in your repo, import it here instead of timm
def _build_model(num_classes=NUM_CLASSES, model_name=MODEL_NAME):
    try:
        # Example: from scripts.eval_binary import build_model  # if you have one
        # from scripts.eval_binary import build_model
        # return build_model(num_classes=num_classes, model_name=model_name)
        import timm
        return timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    except Exception as e:
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
    try:
        model.load_state_dict(sd, strict=True)
    except Exception:
        # Try stripping a leading module prefix (e.g., "model." or "net.")
        new_sd = {}
        for k,v in (sd.items() if isinstance(sd, dict) else []):
            new_sd[k.split(".",1)[-1]] = v
        if new_sd:
            model.load_state_dict(new_sd, strict=False)
        else:
            raise
    return model

def _maybe_load_torchscript(ckpt):
    try:
        ts = torch.jit.load(ckpt, map_location="cpu")
        ts.eval()
        return ts
    except Exception:
        return None

def predict_probs(image_path: str, ckpt: str, device: str):
    """
    Returns: {"Healthy": float, "Unhealthy": float}
    Accepts both (image, ckpt, device) and (ckpt, image, device) ordering at the router level.
    """
    if not os.path.exists(image_path):
        raise RuntimeError(f"Image not found: {image_path}")
    if not os.path.exists(ckpt):
        raise RuntimeError(f"Checkpoint not found: {ckpt}")

    # Try eager model first, else TorchScript
    try_eager = True
    model = None
    if try_eager:
        try:
            model = _build_model(NUM_CLASSES, MODEL_NAME)
            model = _restore(model, ckpt)
            model.eval()
        except Exception:
            model = None

    if model is None:
        model = _maybe_load_torchscript(ckpt)
        if model is None:
            raise RuntimeError("Binary: could not build from ckpt (state_dict) and ckpt is not TorchScript.")

    x = _load_image(image_path)
    x, dev = _to_device(x, device)
    model.to(dev)

    with torch.no_grad():
        y = model(x)  # (1,2) logits or (1,1) logit
        if isinstance(y,(list,tuple)):
            y = y[0]
        y = y.squeeze()

        if y.ndim == 1 and y.numel() == 2:
            p = torch.softmax(y, dim=-1)
            pH, pU = float(p[0].item()), float(p[1].item())
        elif y.ndim == 0 or (y.ndim == 1 and y.numel() == 1):
            # single logit = "Unhealthy" probability via sigmoid
            pU = float(torch.sigmoid(y).item())
            pH = 1.0 - pU
        else:
            raise RuntimeError(f"Unexpected binary output shape: {tuple(y.shape) if hasattr(y,'shape') else type(y)}")

    # Renormalize defensively
    s = pH + pU
    if s <= 0:
        pH, pU = 0.5, 0.5
    else:
        pH, pU = pH/s, pU/s

    return {"Healthy": pH, "Unhealthy": pU}
