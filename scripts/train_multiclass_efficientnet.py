# scripts/train_multiclass_efficientnet.py
import argparse, json, os, pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def build_transforms(train: bool = True):
    if train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.05, 0.05, 0.05, 0.02),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            # Augment **only** in training
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

class LitEffNet(pl.LightningModule):
    def __init__(self, arch, num_classes, lr, weight_decay, class_weights=None, freeze_epochs=2):
        super().__init__()
        self.save_hyperparameters()
        self.model = timm.create_model(arch, pretrained=True, num_classes=num_classes)

        # ---- Optim schedule
        self.freeze_epochs = freeze_epochs
        self.lr_head = lr
        self.lr_body = lr / 10.0
        self.weight_decay = weight_decay

        # ---- Metrics
        self.val_f1  = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.val_acc = MulticlassAccuracy(num_classes=num_classes, average="micro")

        # ---- Loss: register class weights as buffer so it moves with the module
        if class_weights is None:
            class_weights = torch.ones(num_classes, dtype=torch.float32)
        self.register_buffer("class_weights", class_weights.float())
        self.label_smoothing = 0.05

    def on_train_epoch_start(self):
        # Freeze backbone for first N epochs; then unfreeze all
        if self.current_epoch < self.freeze_epochs:
            for n, p in self.model.named_parameters():
                p.requires_grad = any(k in n for k in ["classifier", "fc", "head"])
        else:
            for p in self.model.parameters():
                p.requires_grad = True

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        head_params, body_params = [], []
        for n, p in self.model.named_parameters():
            if any(k in n for k in ["classifier", "fc", "head"]):
                head_params.append(p)
            else:
                body_params.append(p)
        opt = torch.optim.AdamW(
            [
                {"params": head_params, "lr": self.lr_head},
                {"params": body_params, "lr": self.lr_body},
            ],
            weight_decay=self.weight_decay,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}

    def _ce(self, logits, targets):
        return F.cross_entropy(
            logits, targets,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing
        )

    def training_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = self._ce(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = self._ce(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_f1.update(preds, y)
        self.val_acc.update(preds, y)
        self.log("val_loss", loss, prog_bar=False, on_epoch=True, sync_dist=False)
        return {"loss": loss}

    def on_validation_epoch_end(self):
        f1 = self.val_f1.compute()
        acc = self.val_acc.compute()
        self.log("val_f1", f1, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.val_f1.reset()
        self.val_acc.reset()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir", required=True)
    p.add_argument("--val_dir", required=True)
    p.add_argument("--labels_txt", required=True)
    p.add_argument("--arch", default="efficientnet_b0")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--freeze_epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--class_weights_json", required=True)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--persistent_workers", action="store_true")
    p.add_argument("--device", choices=["cpu", "mps", "cuda"], default="mps")
    p.add_argument("--out_ckpt", default="artifacts/checkpoints/multiclass/best.pt")
    p.add_argument("--save_report_dir", default="artifacts/reports")
    p.add_argument("--init_ckpt", default=None)  # plain .pt (exported best) or dict with 'state_dict'
    args = p.parse_args()

    pl.seed_everything(42)

    labels = [l.strip() for l in open(args.labels_txt)]
    num_classes = len(labels)

    # ---- Datasets
    train_ds = datasets.ImageFolder(args.train_dir, transform=build_transforms(train=True))
    val_ds   = datasets.ImageFolder(args.val_dir,   transform=build_transforms(train=False))
    assert len(train_ds.classes) == num_classes == len(labels), \
        "Label mismatch between folders and labels_26.txt"

    # ---- Device (for creating class-weights tensor)
    if args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # ---- Class weights tensor
    cw = torch.tensor(json.load(open(args.class_weights_json)), dtype=torch.float32, device=device)

    # ---- Model (LightningModule)
    model = LitEffNet(
        arch=args.arch,
        num_classes=num_classes,
        lr=args.lr,
        weight_decay=args.weight_decay,
        class_weights=cw,
        freeze_epochs=args.freeze_epochs
    )

    # ---- Optional init from plain/exported .pt (AFTER model is created)
    if args.init_ckpt:
        meta = torch.load(args.init_ckpt, map_location="cpu")
        # accept either {"state_dict":...} or a raw state_dict
        sd = meta.get("state_dict", meta)
        try:
            model.model.load_state_dict(sd, strict=False)
            print(f"[OK] Initialized from {args.init_ckpt}")
        except Exception as e:
            print("[WARN] Could not init from ckpt:", e)

    # ---- Loaders
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, persistent_workers=args.persistent_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, persistent_workers=args.persistent_workers
    )

    # ---- Callbacks
    ckpt_cb = ModelCheckpoint(
        dirpath="artifacts/checkpoints/multiclass",
        filename="multiclass-epoch={epoch:02d}-val_f1={val_f1:.3f}",
        monitor="val_f1", mode="max", save_top_k=1
    )
    lr_cb = LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        callbacks=[ckpt_cb, lr_cb],
        log_every_n_steps=20,
        enable_progress_bar=True,
        precision=32  # keep fp32 on MPS
    )

    # ---- Train
    trainer.fit(model, train_loader, val_loader)

    # ---- Export a plain .pt (state_dict) for inference
    best_ckpt = ckpt_cb.best_model_path
    assert best_ckpt and os.path.exists(best_ckpt), "No best checkpoint saved!"
    state_dict = torch.load(best_ckpt, map_location="cpu")["state_dict"]
    cleaned = {k.replace("model.", "", 1): v for k, v in state_dict.items() if k.startswith("model.")}
    pathlib.Path(args.save_report_dir).mkdir(parents=True, exist_ok=True)
    torch.save(
        {"arch": args.arch, "num_classes": num_classes, "state_dict": cleaned, "labels": labels},
        args.out_ckpt
    )
    print(f"[OK] Saved best Lightning ckpt → {best_ckpt}")
    print(f"[OK] Exported plain weights  → {args.out_ckpt}")

if __name__ == "__main__":
    main()
