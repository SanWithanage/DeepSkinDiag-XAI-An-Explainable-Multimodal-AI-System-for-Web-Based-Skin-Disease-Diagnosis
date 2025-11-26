import os, argparse, json, time
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def make_tfms(img_size):
    train_tfms = transforms.Compose([
        transforms.Resize(int(img_size*1.15)),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize(int(img_size*1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tfms, eval_tfms

def class_weights_from_imagefolder(ds):
    # For CrossEntropyLoss with 2 logits (Healthy=0, Unhealthy=1)
    # weight[i] is applied to samples of class i
    counts = [0,0]
    for _, y in ds.samples:
        counts[y] += 1
    total = sum(counts)
    # inverse frequency (stable) — normalize to mean=1.0
    inv = [total/max(1,c) for c in counts]
    mean = sum(inv)/2.0
    w = [v/mean for v in inv]
    return torch.tensor(w, dtype=torch.float32), counts

class LitEffB0(pl.LightningModule):
    def __init__(self, lr, max_epochs, freeze_epochs, class_weights):
        super().__init__()
        self.save_hyperparameters()
        self.model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=2)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.acc = BinaryAccuracy()
        self.f1  = BinaryF1Score()

        # freeze backbone for first N epochs (unfreeze later)
        for n,p in self.model.named_parameters():
            if "classifier" not in n:
                p.requires_grad = False

    def forward(self, x):
        return self.model(x)  # logits (N,2)

    def _shared_step(self, batch, stage):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        # convert to prob of class 1 (Unhealthy) for binary metrics
        p_unhealthy = torch.softmax(logits, dim=1)[:,1]
        y_bool = (y == 1)
        acc = self.acc(p_unhealthy, y_bool)
        f1  = self.f1(p_unhealthy, y_bool)
        self.log(f"{stage}_loss", loss, prog_bar=(stage!="train"))
        self.log(f"{stage}_acc",  acc,  prog_bar=(stage!="train"))
        self.log(f"{stage}_f1",   f1,   prog_bar=(stage!="train"))
        return loss

    def training_step(self, batch, _):
        # unfreeze after freeze_epochs
        if self.current_epoch == self.hparams.freeze_epochs:
            for p in self.model.parameters():
                p.requires_grad = True
        return self._shared_step(batch, "train")

    def validation_step(self, batch, _):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        # slightly higher LR on classifier head
        head = []
        body = []
        for n,p in self.model.named_parameters():
            (head if "classifier" in n else body).append(p)
        opt = torch.optim.AdamW(
            [{"params": body, "lr": self.hparams.lr},
             {"params": head, "lr": self.hparams.lr*2.0}],
            weight_decay=1e-4
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams.max_epochs)
        return {"optimizer": opt, "lr_scheduler": sch}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--freeze_epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    torch.set_float32_matmul_precision("high")

    train_tfms, eval_tfms = make_tfms(args.img_size)
    train_dir = os.path.join(args.data_root, "train")
    val_dir   = os.path.join(args.data_root, "val")

    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds   = datasets.ImageFolder(val_dir,   transform=eval_tfms)

    class_weights, counts = class_weights_from_imagefolder(train_ds)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    out_dir = Path("artifacts/checkpoints/binary"); out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path("artifacts/reports"); reports_dir.mkdir(parents=True, exist_ok=True)

    model = LitEffB0(lr=args.lr, max_epochs=args.epochs,
                     freeze_epochs=args.freeze_epochs, class_weights=class_weights)

    ckpt = ModelCheckpoint(
        dirpath=str(out_dir),
        filename="binary-{epoch:02d}-val_acc={val_acc:.3f}",
        save_top_k=1, monitor="val_acc", mode="max"
    )
    lrmon = LearningRateMonitor(logging_interval="epoch")

    # Select device automatically (M1 → mps)
    accelerator = "auto"
    devices = 1

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        precision="32-true",
        accelerator=accelerator,
        devices=devices,
        callbacks=[ckpt, lrmon],
        log_every_n_steps=20
    )

    trainer.fit(model, train_loader, val_loader)

    # Export plain state_dict for inference
    best_ckpt = ckpt.best_model_path
    m = LitEffB0.load_from_checkpoint(best_ckpt, lr=args.lr,
        max_epochs=args.epochs, freeze_epochs=args.freeze_epochs,
        class_weights=class_weights)
    torch.save(m.model.state_dict(), out_dir.joinpath("best.pt"))

    # Write tiny metrics file
    stamp = time.strftime("%Y%m%d-%H%M%S")
    metrics_txt = reports_dir.joinpath(f"binary_train_summary_{stamp}.txt")
    with open(metrics_txt, "w") as f:
        f.write(f"Best checkpoint: {best_ckpt}\n")
        f.write(f"Train counts  Healthy={counts[0]}  Unhealthy={counts[1]}\n")
        f.write("Target: ≥0.95 val accuracy\n")
    print(f"Saved best.pt and summary → {out_dir} / {metrics_txt}")

if __name__ == "__main__":
    main()
