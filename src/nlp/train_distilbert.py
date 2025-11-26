import os, json, argparse, numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import f1_score
import evaluate

def read_labels(p):
    labs=[l.strip() for l in open(p) if l.strip()]
    if len(labs)!=26: raise ValueError(f"Expected 26 labels, got {len(labs)}")
    return labs

def load_and_check(t,v,te,valid):
    files={}
    if t: files["train"]=t
    if v: files["validation"]=v
    if te: files["test"]=te
    ds=load_dataset("csv", data_files=files)
    def chk(b):
        bad=[y for y in b["label"] if y not in valid]
        if bad: raise ValueError(f"Found labels not in labels_26.txt: {sorted(set(bad))[:5]}")
        return b
    for k in ds.keys(): ds[k]=ds[k].map(chk, batched=True)
    return ds

def tokenize(ds, tok, l2i, max_len):
    # IMPORTANT: remove original 'label' string column after mapping!
    def _t(batch):
        enc = tok(batch["text"], truncation=True, padding=True, max_length=max_len)
        enc["labels"] = [l2i[y] for y in batch["label"]]  # ints
        return enc
    out={}
    for split, split_ds in ds.items():
        out[split] = split_ds.map(_t, batched=True, remove_columns=split_ds.column_names)
    return out

def metrics():
    acc = evaluate.load("accuracy")
    def fn(pred):
        logits, labels = pred
        preds = np.argmax(logits, axis=-1)
        out = acc.compute(predictions=preds, references=labels)
        out["macro_f1"] = float(f1_score(labels, preds, average="macro"))
        return out
    return fn

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--labels_txt", default="artifacts/labels_26.txt")
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--test_csv")
    ap.add_argument("--out_dir", default="artifacts/checkpoints/symptom_bert")
    ap.add_argument("--report_dir", default="artifacts/reports")
    ap.add_argument("--model_name", default="distilbert-base-uncased")
    ap.add_argument("--max_len", type=int, default=192)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--train_bs", type=int, default=16)
    ap.add_argument("--eval_bs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    a=ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(a.report_dir, exist_ok=True)

    labels = read_labels(a.labels_txt)
    id2label={i:l for i,l in enumerate(labels)}
    label2id={l:i for i,l in enumerate(labels)}

    ds = load_and_check(a.train_csv, a.val_csv, a.test_csv, set(labels))
    tok = AutoTokenizer.from_pretrained(a.model_name)
    ds_tok = tokenize(ds, tok, label2id, a.max_len)

    cfg = AutoConfig.from_pretrained(a.model_name, num_labels=len(labels), id2label=id2label, label2id=label2id)
    model = AutoModelForSequenceClassification.from_pretrained(a.model_name, config=cfg)

    # Minimal args (compatible across versions)
    ta = TrainingArguments(
        output_dir=os.path.join(a.out_dir,"hf_runs"),
        learning_rate=a.lr,
        per_device_train_batch_size=a.train_bs,
        per_device_eval_batch_size=a.eval_bs,
        num_train_epochs=a.epochs,
        weight_decay=0.01,
        seed=a.seed,
    )

    tr = Trainer(
        model=model,
        args=ta,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok.get("validation"),
        tokenizer=tok,  # fine even with warning
        compute_metrics=metrics()
    )

    tr.train()
    val_metrics = tr.evaluate(ds_tok["validation"]) if "validation" in ds_tok else {}
    test_metrics = tr.evaluate(ds_tok["test"]) if "test" in ds_tok else {}

    model.save_pretrained(a.out_dir)
    tok.save_pretrained(a.out_dir)
    with open(os.path.join(a.out_dir, "labels.json"), "w") as f:
        json.dump({"labels": labels}, f, indent=2)

    report = {
        "val": {k: float(v) for k,v in val_metrics.items()},
        "test": {k: float(v) for k,v in test_metrics.items()},
        "config": {
            "model_name": a.model_name, "max_len": a.max_len, "epochs": a.epochs,
            "lr": a.lr, "train_bs": a.train_bs, "eval_bs": a.eval_bs, "seed": a.seed
        }
    }
    rep_path=os.path.join(a.report_dir,"symptom_bert_metrics.json")
    with open(rep_path,"w") as f: json.dump(report, f, indent=2)

    mf1 = report["val"].get("macro_f1", -1.0)
    print(f"[VAL] macro-F1 = {mf1:.4f}  (Target ≥ 0.60)")
    print(f"Saved model+tokenizer → {a.out_dir}")
    print(f"Saved labels.json + metrics → {rep_path}")

if __name__=="__main__":
    main()
