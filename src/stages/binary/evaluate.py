import os, argparse, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import timm
import numpy as np
import matplotlib.pyplot as plt

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def eval_tfms(img_size):
    return transforms.Compose([
        transforms.Resize(int(img_size*1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def load_model(ckpt_path):
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
    sd = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--split", choices=["val","test"], default="val")
    ap.add_argument("--ckpt", default="artifacts/checkpoints/binary/best.pt")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    ds = datasets.ImageFolder(os.path.join(args.data_root, args.split), transform=eval_tfms(args.img_size))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = load_model(args.ckpt)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)

    y_true, y_score = [], []
    with torch.no_grad():
        for x,y in dl:
            x = x.to(device)
            logits = model(x)
            p = torch.softmax(logits, dim=1)[:,1]   # prob(Unhealthy)
            y_true.append(y.cpu().numpy())
            y_score.append(p.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_score = np.concatenate(y_score)
    y_pred = (y_score >= 0.5).astype(np.int64)

    # metrics
    report = classification_report(y_true, y_pred, target_names=["Healthy","Unhealthy"], digits=4)
    cm = confusion_matrix(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = float("nan")

    # save brief report + images
    reports_dir = Path("artifacts/reports"); reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    txt = reports_dir / f"binary_{args.split}_report_{stamp}.txt"
    with open(txt, "w") as f:
        f.write(report + "\n")
        f.write(f"ROC-AUC: {auc:.4f}\n")

        # Router helper thresholds
        # High-confidence Healthy gate: require p(Unhealthy) <= 0.05 → p(Healthy) >= 0.95
        # (You’ll use this in the router to skip multiclass.)
        f.write("Router thresholds:\n")
        f.write("  healthy_high_confidence: p_unhealthy <= 0.05 (i.e., p_healthy >= 0.95)\n")

    # Confusion matrix plot (optional but tiny)
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Binary Confusion Matrix")
    plt.xticks([0,1], ["Healthy","Unhealthy"])
    plt.yticks([0,1], ["Healthy","Unhealthy"])
    for (i,j),v in np.ndenumerate(cm):
        plt.text(j,i,str(v),ha='center',va='center')
    plt.tight_layout()
    cm_path = reports_dir / f"binary_{args.split}_confusion_matrix_{stamp}.png"
    plt.savefig(cm_path, dpi=160)
    print(report)
    print(f"ROC-AUC: {auc:.4f}")
    print(f"Saved → {txt} and {cm_path}")

if __name__ == "__main__":
    main()
