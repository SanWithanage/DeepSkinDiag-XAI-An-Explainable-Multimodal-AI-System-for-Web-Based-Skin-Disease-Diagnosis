# train_text_model.py  — MPS friendly (fp16 + grad accumulation)
import os, json, argparse
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def read_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def load_df(csv_path, labels, label_map=None):
    df = pd.read_csv(csv_path)
    if label_map:
        df["label"] = df["label"].map(lambda x: label_map.get(x, x))
    before = len(df)
    df = df[df["label"].isin(labels)].copy()
    dropped = before - len(df)
    if dropped:
        print(f"⚠️  Dropped {dropped} rows not in canonical labels from {os.path.basename(csv_path)}")
    return df

class TextDS(Dataset):
    def __init__(self, df, tokenizer, label2id, max_len):
        self.texts = df["text"].astype(str).tolist()
        self.y = df["label"].map(label2id).astype(int).tolist()
        self.tok = tokenizer
        self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, padding="max_length",
                       max_length=self.max_len, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.y[i], dtype=torch.long)
        return item

@torch.no_grad()
def evaluate(model, loader, device, precision="fp16"):
    model.eval()
    correct = total = 0
    autocast_dtype = torch.float16 if (precision == "fp16" and device.type in ["mps","cuda"]) else None
    cm = torch.autocast(device_type=device.type, dtype=autocast_dtype) if autocast_dtype else torch.no_grad()
    with cm:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**{k: v for k, v in batch.items() if k != "labels"}).logits
            preds = logits.argmax(dim=1)
            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].size(0)
    return correct / max(1, total)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv",   required=True)
    ap.add_argument("--labels_txt", default="artifacts/labels_26.txt")
    ap.add_argument("--label_map_json", default=None)
    ap.add_argument("--model_name", default="xlm-roberta-base")
    ap.add_argument("--out_dir", default="artifacts/checkpoints/symptom_mbert")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--train_bs", type=int, default=16)
    ap.add_argument("--eval_bs", type=int, default=32)
    ap.add_argument("--grad_accum", type=int, default=1, help="gradient accumulation steps")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--precision", choices=["fp32","fp16"], default=None,
                    help="fp16 recommended on MPS/CUDA (auto-selects fp16 on MPS if not set)")
    args = ap.parse_args()

    # device + precision
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.precision is None:
        args.precision = "fp16" if device.type in ["mps","cuda"] else "fp32"
    print(f"Device: {device} | Precision: {args.precision}")

    # labels
    labels = read_labels(args.labels_txt)
    label2id = {n: i for i, n in enumerate(labels)}
    id2label = {i: n for n, i in label2id.items()}

    # optional label map
    label_map = None
    if args.label_map_json and os.path.exists(args.label_map_json):
        with open(args.label_map_json, "r", encoding="utf-8") as f:
            label_map = json.load(f)

    # data
    train_df = load_df(args.train_csv, labels, label_map)
    val_df   = load_df(args.val_csv,   labels, label_map)
    print(f"Train rows: {len(train_df)} | Val rows: {len(val_df)} | Classes: {len(labels)}")

    tok = AutoTokenizer.from_pretrained(args.model_name)
    dtrain = TextDS(train_df, tok, label2id, args.max_len)
    dval   = TextDS(val_df,   tok, label2id, args.max_len)

    # low workers to reduce RAM on macOS
    train_loader = DataLoader(dtrain, batch_size=args.train_bs, shuffle=True, num_workers=0)
    val_loader   = DataLoader(dval,   batch_size=args.eval_bs, shuffle=False, num_workers=0)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(labels), id2label=id2label, label2id=label2id
    ).to(device)

    # optimizer + scheduler
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    grouped = [
        {"params":[p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params":[p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(grouped, lr=args.lr)
    num_train_steps = (len(train_loader) // max(1, args.grad_accum)) * args.epochs
    warmup_steps = int(args.warmup_ratio * num_train_steps)
    def lr_lambda(step):
        if step < warmup_steps and warmup_steps > 0:
            return float(step) / float(max(1, warmup_steps))
        return max(0.0, float(num_train_steps - step) / float(max(1, num_train_steps - warmup_steps)))
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    best_acc, best_path = -1.0, None
    os.makedirs(args.out_dir, exist_ok=True)

    autocast_dtype = torch.float16 if (args.precision == "fp16" and device.type in ["mps","cuda"]) else None

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            cm = torch.autocast(device_type=device.type, dtype=autocast_dtype) if autocast_dtype else torch.enable_grad()
            with cm:
                out = model(**{k: v for k, v in batch.items() if k != "labels"})
                loss = F.cross_entropy(out.logits, batch["labels"], label_smoothing=0.05)

            loss = loss / max(1, args.grad_accum)
            loss.backward()

            if step % max(1, args.grad_accum) == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if device.type == "mps":
                    try:
                        torch.mps.empty_cache()
                    except Exception:
                        pass

            running_loss += loss.item() * max(1, args.grad_accum)

        val_acc = evaluate(model, val_loader, device, precision=args.precision)
        print(f"Epoch {epoch:02d} | train_loss: {running_loss/len(train_loader):.4f} | val_acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            model.save_pretrained(args.out_dir)
            tok.save_pretrained(args.out_dir)
            best_path = args.out_dir
            print(f"💾 Saved new best to {best_path} (val_acc={best_acc:.4f})")

    print(f"✅ Done. Best val_acc={best_acc:.4f}. Model at: {best_path}")

if __name__ == "__main__":
    main()
