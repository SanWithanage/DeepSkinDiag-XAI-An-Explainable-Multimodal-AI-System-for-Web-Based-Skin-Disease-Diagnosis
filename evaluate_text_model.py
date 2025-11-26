# evaluate_text_model.py
# Evaluate a Hugging Face text classifier (DistilBERT/BERT) on symptom CSVs.
# CSV must have columns: text,label  (label can be class name; we map to IDs via labels_26.txt)
# Usage:
#   python evaluate_text_model.py \
#       --text_ckpt artifacts/checkpoints/symptom_bert_hf \
#       --test_csv  /Users/sandunwithanage/Documents/Data001/Symptoms_Data/symptom_test.csv \
#       --labels_txt artifacts/labels_26.txt
#
# Optional:
#   --label_map_json path/to/map.json   # custom mapping file { "Basal_Cell_Carcinoma": "Skin_Cancer", ... }
#   --save_probs_csv out.csv            # writes per-example probabilities & predictions

import argparse, os, json
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --------- helpers ---------
def read_labels(labels_txt_path: str):
    with open(labels_txt_path, "r", encoding="utf-8") as f:
        names = [ln.strip() for ln in f if ln.strip()]
    return names

def load_label_map(label_map_json: str | None):
    if not label_map_json:
        return None
    if not os.path.exists(label_map_json):
        raise FileNotFoundError(f"--label_map_json not found: {label_map_json}")
    with open(label_map_json, "r", encoding="utf-8") as f:
        return json.load(f)

# A small built-in mapping for common fine-grained names -> canonical 26-class names.
# Edit to match your exact 26-class taxonomy if needed.
DEFAULT_NAME_MAP = {
    "Basal_Cell_Carcinoma": "Skin_Cancer",
    "Cellulitis": "Bacterial_Infections",
    "Contact_Dermatitis": "Eczema",
    "Folliculitis": "Bacterial_Infections",
    "Herpes_Simplex": "Herpes",
    "Impetigo": "Bacterial_Infections",
    "Keloid": "Vascular_Tumors",
    "Keratosis_Pilaris": "Seborrh_Keratoses",
    "Lichen_Planus": "Lichen",
    "Lupus_Rash": "Lupus",
    # add/adjust more as needed...
}

class SymptomDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, label2id: dict[str, int], max_len: int = 64):
        # ensure strings
        self.texts = df["text"].astype(str).tolist()
        self.labels = [label2id[x] for x in df["label"].tolist()]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        enc["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return enc

def evaluate(model, dataloader, device, save_probs_path: str | None = None, id2label: dict[int, str] | None = None):
    model.eval()
    correct, total = 0, 0
    rows = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            labels = batch["labels"].to(device)
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            out = model(**inputs)
            probs = F.softmax(out.logits, dim=1)
            preds = probs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if save_probs_path:
                # collect per-row results
                for i in range(labels.size(0)):
                    pred_id = int(preds[i].item())
                    true_id = int(labels[i].item())
                    rows.append({
                        "true_id": true_id,
                        "true_label": id2label[true_id] if id2label else true_id,
                        "pred_id": pred_id,
                        "pred_label": id2label[pred_id] if id2label else pred_id,
                        "max_prob": float(probs[i, pred_id].item()),
                    })

    acc = correct / max(1, total)
    if save_probs_path and rows:
        pd.DataFrame(rows).to_csv(save_probs_path, index=False)
        print(f"💾 Saved per-row predictions to: {save_probs_path}")
    return acc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text_ckpt", required=True,
                    help="HF directory containing model.safetensors, config.json, tokenizer files")
    ap.add_argument("--test_csv", required=True,
                    help="CSV with columns: text,label (label as class name)")
    ap.add_argument("--labels_txt", default="artifacts/labels_26.txt",
                    help="One class name per line; order must match your 26-class image model")
    ap.add_argument("--label_map_json", default=None,
                    help="Optional JSON mapping of fine-grained labels to canonical 26-class names")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--save_probs_csv", default=None,
                    help="Optional path to save per-example predictions/probs")
    args = ap.parse_args()

    # 1) load canonical class list (26-class order)
    if not os.path.exists(args.labels_txt):
        raise FileNotFoundError(f"labels file not found: {args.labels_txt}")
    class_names = read_labels(args.labels_txt)
    id2label = {i: n for i, n in enumerate(class_names)}
    label2id = {n: i for i, n in enumerate(class_names)}
    canon = set(class_names)

    # 2) load CSV
    if not os.path.exists(args.test_csv):
        raise FileNotFoundError(f"test_csv not found: {args.test_csv}")
    df = pd.read_csv(args.test_csv)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns")

    # 3) apply label mapping (custom JSON overrides default)
    label_map = load_label_map(args.label_map_json) or DEFAULT_NAME_MAP
    if label_map:
        df["label"] = df["label"].map(lambda x: label_map.get(x, x))

    # 4) drop rows whose labels are not in the 26-class set
    unknown = sorted(set(df["label"]) - canon)
    if unknown:
        before = len(df)
        df = df[df["label"].isin(canon)].copy()
        after = len(df)
        print(f"⚠️ Dropped {before-after} rows with labels not in 26-class list.")
        print(f"   First unknowns (for mapping later): {unknown[:10]}")
    if len(df) == 0:
        raise ValueError("No evaluable rows after mapping/filtering. Update your label map to cover CSV labels.")

    # 5) device + model
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.text_ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.text_ckpt,
        num_labels=len(class_names),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True  # tolerates a different head shape; HF will re-init the head
    ).to(device)

    # 6) dataloader
    ds = SymptomDataset(df, tokenizer, label2id, max_len=args.max_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    # 7) evaluate
    print(f"🧪 Evaluating on {len(df)} rows | device={device} | classes={len(class_names)}")
    acc = evaluate(model, dl, device, save_probs_path=args.save_probs_csv, id2label=id2label)
    print(f"\n✅ Text-only model accuracy on test set: {acc:.4f}")

if __name__ == "__main__":
    main()
