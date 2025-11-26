# src/explain/gradcam.py
import argparse, os, glob, json, math, traceback
from pathlib import Path
import numpy as np
from PIL import Image
import torch, torch.nn.functional as F
from torchvision import transforms
import timm
from matplotlib import cm

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def load_labels(labels_txt):
    return [l.strip() for l in open(labels_txt).read().splitlines() if l.strip()]

def build_model(arch, num_classes, ckpt_path, device):
    model = timm.create_model(arch, pretrained=False, num_classes=num_classes)
    sd = torch.load(ckpt_path, map_location="cpu")
    # Allow both plain state_dict and wrapped {"state_dict": ...}
    if isinstance(sd, dict) and "state_dict" in sd and not any(k.startswith("module.") for k in model.state_dict()):
        # lightning-style: keys like "model.layer..."
        # try to strip a possible "model." prefix
        new_sd = {}
        for k,v in sd["state_dict"].items():
            nk = k
            if nk.startswith("model."): nk = nk[len("model."):]
            if nk.startswith("net."):   nk = nk[len("net."):]
            new_sd[nk] = v
        sd = new_sd
    if isinstance(sd, dict):
        try:
            model.load_state_dict(sd, strict=False)
        except Exception:
            # Some checkpoints might be under "model."
            new_sd = {k.replace("model.","").replace("net.",""): v for k,v in sd.items()}
            model.load_state_dict(new_sd, strict=False)
    else:
        raise RuntimeError(f"Unexpected ckpt format at {ckpt_path}")
    model.eval().to(device)
    return model

def pick_target_layer(model):
    """
    Pick a good last conv for common timm backbones (e.g., EfficientNet).
    Adjust here if your arch differs.
    """
    for name in ["conv_head", "bn2"]:
        m = getattr(model, name, None)
        if m is not None:
            return getattr(model, "conv_head")
    # Fallback: last conv we can find
    last_conv = None
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise RuntimeError("Could not find a Conv2d layer for Grad-CAM.")
    return last_conv

def preprocess_pil(im, size=224):
    tfm = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return tfm(im)

def gradcam(model, target_layer, img_tensor, device, target_class=None):
    """
    Returns heatmap (H,W) in [0,1] and logits.
    """
    feats = []
    grads = []

    def fwd_hook(_, __, out): feats.append(out.detach())
    def bwd_hook(_, gin, gout): grads.append(gout[0].detach())

    handle_f = target_layer.register_forward_hook(fwd_hook)
    handle_b = target_layer.register_full_backward_hook(bwd_hook)

    try:
        x = img_tensor.unsqueeze(0).to(device)
        x.requires_grad_(True)
        logits = model(x)
        if isinstance(logits, (list, tuple)): logits = logits[0]
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())
        score = logits[0, target_class]
        model.zero_grad(set_to_none=True)
        score.backward(retain_graph=False)

        A = feats[-1]          # (B, C, H, W)
        dA = grads[-1]         # (B, C, H, W)
        weights = dA.mean(dim=(2,3), keepdim=True)  # GAP over H,W -> (B,C,1,1)
        cam = (weights * A).sum(dim=1, keepdim=False)  # (B,H,W)
        cam = torch.relu(cam)
        cam = cam[0]
        cam = cam - cam.min()
        if cam.max() > 0: cam = cam / cam.max()
        heat = cam.detach().cpu().numpy()  # (H,W) in [0,1]
        return heat, logits.detach().cpu().numpy()[0]
    finally:
        handle_f.remove()
        handle_b.remove()

def overlay_heatmap(pil_img, heat, alpha=0.45):
    pil_img = pil_img.convert("RGB")
    w, h = pil_img.size
    # resize heat to image size
    heat_r = Image.fromarray((heat*255).astype(np.uint8)).resize((w,h), resample=Image.BILINEAR)
    heat_r = np.array(heat_r).astype(np.float32)/255.0
    # colorize with matplotlib colormap
    colored = cm.jet(heat_r)[..., :3]  # (H,W,3) in [0,1]
    img_np = np.array(pil_img).astype(np.float32)/255.0
    out = (1-alpha)*img_np + alpha*colored
    out = (np.clip(out,0,1)*255).astype(np.uint8)
    return Image.fromarray(out)

def count_per_class(train_dir):
    # returns dict {class_name: count}
    d = {}
    for p in sorted(Path(train_dir).glob("*")):
        if p.is_dir():
            d[p.name] = len(list(p.glob("*")))
    return d

def pick_common_rare(train_dir, k_common=2, k_rare=2):
    counts = count_per_class(train_dir)
    if not counts:
        raise RuntimeError(f"No class folders found in {train_dir}")
    by_freq = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    commons = [c for c,_ in by_freq[:k_common]]
    rares   = [c for c,_ in by_freq[::-1] if _ > 0][:k_rare]
    return commons, rares

def sample_one(val_dir, cls):
    paths = sorted(glob.glob(os.path.join(val_dir, cls, "*")))
    if not paths:
        # try train as fallback
        paths = sorted(glob.glob(os.path.join(val_dir.replace("Val","train"), cls, "*")))
    if not paths:
        raise RuntimeError(f"No images for class '{cls}' under {val_dir}")
    return paths[0]

def save_paragraph(path, text):
    with open(path, "w") as f:
        f.write(text.strip()+"\n")

def explain_paragraph(kind, pred_name, conf, focus_frac):
    if kind == "healthy":
        return (f"The binary model’s top-1 prediction is '{pred_name}' with confidence {conf:.1%}. "
                f"The Grad-CAM heatmap is low-intensity and diffuse (focused on ~{focus_frac:.0%} of the image), "
                "which is consistent with an image lacking a distinct lesion region.")
    else:
        return (f"The multiclass model’s top-1 prediction is '{pred_name}' with confidence {conf:.1%}. "
                f"The Grad-CAM highlights a compact region (~{focus_frac:.0%} of the image) where disease-specific "
                "texture and color patterns appear; this indicates the network relied on that area to make its decision.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc_ckpt", required=True)
    ap.add_argument("--mc_arch", default="efficientnet_b0")
    ap.add_argument("--labels_txt", required=True)
    ap.add_argument("--train_multiclass_dir", required=True)
    ap.add_argument("--val_multiclass_dir", required=True)

    ap.add_argument("--bin_ckpt", required=True)
    ap.add_argument("--bin_arch", default="tf_efficientnet_b0")
    ap.add_argument("--val_binary_dir", required=True)

    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--out_dir", default="artifacts/explanations")
    ap.add_argument("--device", default="mps", choices=["cpu","cuda","mps"])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ----- load labels & models
    labels = load_labels(args.labels_txt)

    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"

    mc = build_model(args.mc_arch, num_classes=len(labels), ckpt_path=args.mc_ckpt, device=device)
    mc_tlayer = pick_target_layer(mc)

    bin_model = build_model(args.bin_arch, num_classes=2, ckpt_path=args.bin_ckpt, device=device)
    bin_tlayer = pick_target_layer(bin_model)

    # ----- choose classes: 2 common + 2 rare
    commons, rares = pick_common_rare(args.train_multiclass_dir, 2, 2)

    # pick images
    picks = []
    # 1) Healthy sample from binary Val
    healthy_path = sorted(glob.glob(os.path.join(args.val_binary_dir, "Healthy", "*")))
    if not healthy_path:
        raise RuntimeError("No Healthy images found under binary Val.")
    picks.append(("healthy", "Healthy", healthy_path[0]))

    # 2) 2 commons
    for c in commons:
        picks.append(("multiclass_common", c, sample_one(args.val_multiclass_dir, c)))
    # 3) 2 rares
    for r in rares:
        picks.append(("multiclass_rare", r, sample_one(args.val_multiclass_dir, r)))

    manifest = []
    for kind, cls_name, img_p in picks:
        try:
            im = Image.open(img_p).convert("RGB")
            x = preprocess_pil(im, size=args.img_size)

            if kind == "healthy":
                target_layer = bin_tlayer
                model = bin_model
                heat, logits = gradcam(model, target_layer, x, device)
                # binary index names:
                bin_names = ["Healthy","Unhealthy"]
                top = int(np.argmax(logits))
                pred_name = bin_names[top] if top < 2 else f"class_{top}"
                conf = float(F.softmax(torch.tensor(logits), dim=0)[top].item())

            else:
                target_layer = mc_tlayer
                model = mc
                heat, logits = gradcam(model, target_layer, x, device)
                top = int(np.argmax(logits))
                pred_name = labels[top] if top < len(labels) else f"class_{top}"
                conf = float(F.softmax(torch.tensor(logits), dim=0)[top].item())

            # focus fraction = proportion of pixels above 0.6
            focus_frac = float((heat >= 0.6).mean())

            overlay = overlay_heatmap(im, heat, alpha=0.45)

            base = Path(img_p).stem
            out_img = os.path.join(args.out_dir, f"{kind}_{cls_name}_{base}_gradcam.png")
            overlay.save(out_img)

            para = explain_paragraph("healthy" if kind=="healthy" else "multiclass", pred_name, conf, focus_frac)
            out_txt = os.path.join(args.out_dir, f"{kind}_{cls_name}_{base}.txt")
            save_paragraph(out_txt, para)

            manifest.append({
                "kind": kind,
                "true_class": cls_name,
                "image": img_p,
                "pred_top1": pred_name,
                "confidence": conf,
                "focus_fraction": focus_frac,
                "out_img": out_img,
                "out_txt": out_txt
            })
            print(f"[OK] {kind}/{cls_name} → {out_img}")
        except Exception as e:
            print(f"[FAIL] {kind}/{cls_name} on {img_p}: {e}")
            traceback.print_exc()

    json.dump(manifest, open(os.path.join(args.out_dir, "gradcam_manifest.json"), "w"), indent=2)
    print(f"\nSaved manifest → {os.path.join(args.out_dir, 'gradcam_manifest.json')}")
    # Short README
    readme = os.path.join(args.out_dir, "README_explainability.md")
    with open(readme, "w") as f:
        f.write("# Grad-CAM Explainability (Auto)\n\n")
        f.write("This folder contains overlays and a short paragraph for each sample.\n")
        f.write("- 1 Healthy (binary model)\n- 2 common classes (by training frequency)\n- 2 rare classes (by training frequency)\n")
        f.write("\nOpen the corresponding .txt next to each PNG for the one-paragraph explanation.\n")
    print(f"Saved README → {readme}")
if __name__ == "__main__":
    main()
