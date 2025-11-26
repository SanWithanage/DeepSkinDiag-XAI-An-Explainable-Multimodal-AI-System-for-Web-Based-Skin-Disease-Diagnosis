import argparse, json, os, pathlib
import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

IMAGENET_MEAN=(0.485,0.456,0.406)
IMAGENET_STD=(0.229,0.224,0.225)

def val_tfms():
    import torchvision.transforms as T
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def build_model(arch, num_classes, state_dict):
    import timm, torch.nn as nn
    m = timm.create_model(arch, pretrained=False, num_classes=num_classes)
    m.load_state_dict(state_dict, strict=True)
    m.eval()
    return m

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)  # artifacts/checkpoints/multiclass/best.pt
    p.add_argument("--labels_txt", required=True)
    p.add_argument("--val_dir", required=True)
    p.add_argument("--out_dir", default="artifacts/reports")
    p.add_argument("--tta", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    labels = [l.strip() for l in open(args.labels_txt)]
    meta = torch.load(args.ckpt, map_location="cpu")
    arch = meta["arch"]; num_classes = meta["num_classes"]; state_dict = meta["state_dict"]

    model = build_model(arch, num_classes, state_dict)

    ds = datasets.ImageFolder(args.val_dir, transform=val_tfms())
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)

    
    def predict_logits(model, x):
        with torch.no_grad():
            return model(x).cpu()

    y_true, y_pred = [], []
    for x,y in loader:
        if args.tta:
            x_flip = torch.flip(x, dims=[3])  # horizontal flip
            logits = predict_logits(model, x) + predict_logits(model, x_flip)
        else:
            logits = predict_logits(model, x)
        pred = logits.argmax(1).numpy().tolist()
        y_true += y.numpy().tolist()
        y_pred += pred

    # Reports
    rep = classification_report(y_true, y_pred, target_names=labels, digits=4, output_dict=False)
    ts = time_str = __import__("time").strftime("%Y%m%d-%H%M%S")
    txt_path = os.path.join(args.out_dir, f"multiclass_val_report_{ts}.txt")
    with open(txt_path, "w") as f:
        f.write(rep)
    print(f"[OK] Wrote report → {txt_path}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    fig = plt.figure(figsize=(14,12))
    sns.heatmap(cm, annot=False, cmap="Blues")
    plt.title("Confusion Matrix (Val)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    png_path = os.path.join(args.out_dir, "multiclass_confusion_matrix.png")
    plt.tight_layout(); plt.savefig(png_path, dpi=200); plt.close(fig)
    print(f"[OK] Wrote confusion matrix → {png_path}")

    # Macro-F1 / Top-1 quick summary
    import numpy as np
    from sklearn.metrics import f1_score, accuracy_score
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    acc = accuracy_score(y_true, y_pred)
    print(f"[VAL] macroF1={macro_f1:.4f}  top1_acc={acc:.4f}")

if __name__ == "__main__":
    main()
