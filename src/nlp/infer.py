import json, argparse, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--ckpt", default="artifacts/checkpoints/symptom_bert")
    ap.add_argument("--topk", type=int, default=5)
    args=ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(args.ckpt)
    labels = json.load(open(f"{args.ckpt}/labels.json"))["labels"]

    enc = tok(args.text, return_tensors="pt", truncation=True, max_length=192)
    with torch.no_grad():
        logits = model(**enc).logits[0].cpu().numpy()
    probs = softmax(logits).tolist()
    idx = np.argsort(probs)[::-1][:args.topk]
    topk = [{"label": labels[i], "prob": float(probs[i])} for i in idx]
    print(json.dumps({"text": args.text, "topk": topk, "probs": probs}, indent=2))

if __name__=="__main__":
    main()
