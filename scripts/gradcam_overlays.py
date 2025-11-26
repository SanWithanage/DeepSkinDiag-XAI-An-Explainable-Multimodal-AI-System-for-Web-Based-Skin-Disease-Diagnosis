#!/usr/bin/env python3
import argparse, os, sys, json, math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import timm
from matplotlib import cm

# -------- utils
def read_labels(p):
    return [ln.strip() for ln in open(p, "r").read().splitlines() if ln.strip()]

def build_model(arch, num_classes):
    model = timm.create_model(arch, pretrained=False, num_classes=num_classes)
    return model

def load_state_safely(model, ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    # strip common prefixes
    cleaned = {}
    for k, v in sd.items():
        nk = k
        for pref in ("model.", "net.", "module."):
            if nk.startswith(pref):
                nk = nk[len(pref):]
        if nk in model.state_dict() and model.state_dict()[nk].shape == v.shape:
            cleaned[nk] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if len(missing) > 0:
        print(f"[warn] missing keys: {len(missing)}")
    if len(unexpected) > 0:
        print(f"[warn] unexpected keys: {len(unexpected)}")

def find_last_conv(module: torch.nn.Module):
    last = None
    for m in module.modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    if last is None:
        raise RuntimeError("No Conv2d layer found to attach Grad-CAM hooks.")
    return last

def pil_to_tensor(img: Image.Image, size=224):
    tfm = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return tfm(img)

def tensor_to_numpy_image(t: torch.Tensor):
    # t is normalized tensor [3,H,W]
    t = t.detach().cpu()
    mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    x = (t*std + mean).clamp(0,1).permute(1,2,0).numpy()
    x = (x*255.0).astype(np.uint8)
    return x

def make_heatmap_overlay(base_rgb: np.ndarray, cam_01: np.ndarray, alpha=0.45):
    cmap = cm.get_cmap("jet")
    heat = cmap(cam_01)[..., :3]  # drop A
    heat = (heat * 255.0).astype(np.uint8)
    # blend
    # ensure shapes match
    if heat.shape[:2] != base_rgb.shape[:2]:
        raise ValueError("Heatmap and base image size mismatch")
    out = (alpha*heat + (1.0-alpha)*base_rgb).astype(np.uint8)
    return out

# -------- Grad-CAM core
class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activ = None
        self.grads = None
        self.h1 = target_layer.register_forward_hook(self._save_activ)
        self.h2 = target_layer.register_full_backward_hook(self._save_grads)

    def _save_activ(self, m, i, o):
        self.activ = o.detach()

    def _save_grads(self, m, gin, gout):
        # gout is tuple; take grad wrt activations
        self.grads = gout[0].detach()

    def remove(self):
        self.h1.remove(); self.h2.remove()

    def __call__(self, scores: torch.Tensor, class_idx: int):
        # expects you already did a forward pass and have activ/grads after backward
        # scores shape [1, C]
        if scores.ndim != 2 or scores.shape[0] != 1:
            raise ValueError("scores should be logits for a single image [1, num_classes]")
        self.model.zero_grad(set_to_none=True)
        score = scores[0, class_idx]
        score.backward(retain_graph=False)

        A = self.activ[0]          # [K, h, w]
        G = self.grads[0]          # [K, h, w]
        weights = G.mean(dim=(1,2))  # [K]
        cam = (weights[:,None,None] * A).sum(dim=0)  # [h,w]
        cam = torch.relu(cam)
        if torch.max(cam) > 0:
            cam = cam / (torch.max(cam) + 1e-12)
        return cam  # [h,w] in [0,1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Multiclass model checkpoint (.pt)")
    ap.add_argument("--arch", default="efficientnet_b0", help="timm model name")
    ap.add_argument("--labels_txt", required=True)
    ap.add_argument("--out_dir", default="artifacts/reports/explainability")
    ap.add_argument("--device", default="mps", choices=["cpu","cuda","mps"])
    ap.add_argument("images", nargs="+", help="4–6 image paths")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    labels = read_labels(args.labels_txt)
    device = torch.device(args.device if torch.backends.mps.is_available() or args.device != "mps" else "cpu")

    model = build_model(args.arch, num_classes=len(labels))
    load_state_safely(model, args.ckpt)
    model.to(device).eval()

    target_layer = find_last_conv(model)
    cam_engine = GradCAM(model, target_layer)

    saved = 0
    for img_path in args.images:
        if not os.path.isfile(img_path):
            print(f"[skip] not found: {img_path}")
            continue

        img_pil = Image.open(img_path).convert("RGB")
        x = pil_to_tensor(img_pil, size=224)
        x = x.unsqueeze(0).to(device)

        with torch.enable_grad():
            logits = model(x)               # [1, C]
            probs = logits.softmax(dim=1)   # [1, C]
            top_idx = int(probs.argmax(dim=1).item())
            cam_small = cam_engine(logits, top_idx)   # [h,w] in [0,1]

        # upscale CAM to input image size
        H, W = x.shape[2], x.shape[3]
        cam_up = F.interpolate(cam_small[None,None], size=(H,W), mode="bilinear", align_corners=False)[0,0]
        cam_up = cam_up.clamp(0,1).detach().cpu().numpy()

        base_rgb = tensor_to_numpy_image(x[0])
        overlay = make_heatmap_overlay(base_rgb, cam_up, alpha=0.45)

        pred_label = labels[top_idx] if 0 <= top_idx < len(labels) else f"cls_{top_idx}"
        base = Path(img_path).stem
        out_png = os.path.join(args.out_dir, f"{base}__{pred_label}_gradcam.png")
        Image.fromarray(overlay).save(out_png)
        print(f"[saved] {out_png}")
        saved += 1

    cam_engine.remove()
    print(f"Done. Saved {saved} overlays → {args.out_dir}")

if __name__ == "__main__":
    main()
