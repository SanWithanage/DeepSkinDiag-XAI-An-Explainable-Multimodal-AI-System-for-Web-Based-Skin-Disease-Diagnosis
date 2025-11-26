# scripts/fit_temp_generic.py
import os, argparse, json, math
import torch, torch.nn as nn
from torch.optim import LBFGS
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import f1_score
import timm

torch.set_grad_enabled(True)

def read_labels(labels_txt):
    with open(labels_txt, "r") as f:
        labels = [ln.strip() for ln in f if ln.strip()]
    return labels

def build_val_loader(val_dir, img_size=224, batch_size=64, workers=2):
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    ds = datasets.ImageFolder(val_dir, transform=tfm)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return ds, dl

def build_model(arch, num_classes, ckpt_path, device):
    model = timm.create_model(arch, pretrained=False, num_classes=num_classes)
    sd = torch.load(ckpt_path, map_location="cpu")
    # Handle common checkpoint formats
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    # Strip typical prefixes (e.g., 'model.', 'net.', 'module.')
    new_sd = {}
    for k,v in sd.items():
        nk = k
        for pref in ["model.", "net.", "module."]:
            if nk.startswith(pref):
                nk = nk[len(pref):]
        new_sd[nk] = v
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if missing or unexpected:
        print("[warn] load_state_dict non-strict:",
              f"missing={len(missing)} unexpected={len(unexpected)}")
    return model.to(device).eval()

class TempScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.logT = nn.Parameter(torch.zeros(1))  # T = exp(logT) >= 0
    def forward(self, logits):
        return logits / torch.exp(self.logT)

@torch.no_grad()
def collect_logits_labels(model, dl, device):
    all_logits, all_y = [], []
    for x, y in dl:
        x = x.to(device, non_blocking=True)
        logits = model(x)  # raw logits
        all_logits.append(logits.cpu())
        all_y.append(y.cpu())
    return torch.cat(all_logits), torch.cat(all_y)

def expected_calibration_error(probs, y, n_bins=15):
    # probs: (N,C), y: (N,)
    conf, pred = probs.max(1)
    acc = (pred == y).float()
    ece = torch.zeros(1)
    boundaries = torch.linspace(0, 1, steps=n_bins+1)
    for i in range(n_bins):
        l, r = boundaries[i], boundaries[i+1]
        mask = (conf > l) & (conf <= r)
        if mask.any():
            ece += mask.float().mean() * (acc[mask].mean() - conf[mask].mean()).abs()
    return float(ece)

def macro_f1_from_probs(probs, y):
    y_pred = probs.argmax(1).numpy()
    return f1_score(y.numpy(), y_pred, average="macro")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--labels_txt", required=True)
    ap.add_argument("--out_file", default="artifacts/calibration/multiclass_temp.json")
    ap.add_argument("--arch", default="efficientnet_b0")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--device", default="mps")  # mps|cuda|cpu
    ap.add_argument("--ece_bins", type=int, default=15)
    args = ap.parse_args()

    # device
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    if args.device == "mps" and not torch.backends.mps.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    # labels & data
    labels = read_labels(args.labels_txt)
    num_classes = len(labels)
    _, val_loader = build_val_loader(args.val_dir, img_size=args.img_size)

    # model & val logits
    model = build_model(args.arch, num_classes, args.ckpt, device)
    logits, y = collect_logits_labels(model, val_loader, device)

    nll = nn.CrossEntropyLoss()
    # Baseline metrics (no temp)
    base_probs = torch.softmax(logits, dim=1)
    base_ece = expected_calibration_error(base_probs, y, n_bins=args.ece_bins)
    base_nll = float(nll(logits, y))
    base_f1 = macro_f1_from_probs(base_probs, y)

    # Fit T
    scaler = TempScaler()
    opt = LBFGS(scaler.parameters(), lr=0.25, max_iter=50, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        loss = nll(scaler(logits), y)
        loss.backward()
        return loss
    opt.step(closure)

    with torch.no_grad():
        T = float(torch.exp(scaler.logT))
        cal_logits = scaler(logits)
        cal_probs = torch.softmax(cal_logits, dim=1)
        cal_ece = expected_calibration_error(cal_probs, y, n_bins=args.ece_bins)
        cal_nll = float(nll(cal_logits, y))
        cal_f1 = macro_f1_from_probs(cal_probs, y)

    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    with open(args.out_file, "w") as f:
        json.dump({"T": T}, f)

    print(f"T={T:.4f}")
    print(f"NLL: {base_nll:.4f} → {cal_nll:.4f}")
    print(f"ECE: {base_ece:.4f} → {cal_ece:.4f}")
    print(f"macro-F1: {base_f1:.4f} → {cal_f1:.4f}")
    print("Saved:", args.out_file)

    # Success check for your step:
    ok_ece = cal_ece <= 0.08 + 1e-9
    ok_f1 = (base_f1 - cal_f1) <= 0.01 + 1e-9
    print(f"[PASS ECE≤0.08?] {ok_ece}   [PASS ΔF1≤0.01?] {ok_f1}")

if __name__ == "__main__":
    main()
