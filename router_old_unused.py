#!/usr/bin/env python3
import argparse, json, os, sys, glob, subprocess

# --- Defaults (auto-discovery will find actual files)
D = {
  "bin_dir": "artifacts/checkpoints/binary",
  "mc_dir":  "artifacts/checkpoints/multiclass",
  "labels":  "artifacts/labels_26.txt",
  "temp":    "artifacts/calibration/multiclass_temp.json",
  "textdir": "artifacts/checkpoints/symptom_bert_hf",
  "device":  "mps",
  "alpha":   0.50
}

def _pick(patterns):
    for p in patterns:
        hits = glob.glob(p)
        if hits:
            return sorted(hits)[-1]
    return None

def _discover():
    bin_ckpt = _pick([f"{D['bin_dir']}/best.pt",
                      f"{D['bin_dir']}/*.ckpt",
                      f"{D['bin_dir']}/*best*.pt"])
    mc_ckpt  = _pick([f"{D['mc_dir']}/best.pt",
                      f"{D['mc_dir']}/*.pt",
                      f"{D['mc_dir']}/*.ckpt"])
    if not bin_ckpt or not mc_ckpt:
        sys.stderr.write("ERROR: Could not find binary/multiclass checkpoints.\n")
        sys.exit(2)
    if not os.path.exists(D["labels"]):
        sys.stderr.write("ERROR: labels_26.txt missing.\n"); sys.exit(2)
    if not os.path.exists(D["temp"]):
        sys.stderr.write("ERROR: multiclass_temp.json missing.\n"); sys.exit(2)
    return bin_ckpt, mc_ckpt

def _run_router(image_path, symptom_text):
    bin_ckpt, mc_ckpt = _discover()
    cmd = [
        sys.executable, "-m", "src.router",
        "--image", image_path,
        "--binary_ckpt", bin_ckpt,
        "--multiclass_ckpt", mc_ckpt,
        "--multiclass_temp", D["temp"],
        "--labels_txt", D["labels"],
        "--alpha", str(D["alpha"]),
        "--device", D["device"]
    ]
    if symptom_text:
        # Use HF folder; src.router should load tokenizer+model from here
        cmd += ["--text_ckpt", D["textdir"], "--symptom_text", symptom_text]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr or "router failed\n")
        sys.exit(proc.returncode)
    # src.router prints JSON on stdout
    try:
        return json.loads(proc.stdout.strip())
    except Exception:
        # If there was extra logging, try to extract last JSON-looking block
        text = proc.stdout.strip()
        start = text.rfind("{")
        if start != -1:
            try:
                return json.loads(text[start:])
            except Exception:
                pass
        sys.stderr.write("ERROR: Could not parse router JSON.\n")
        print(text)
        sys.exit(3)

def _get(d, path, default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur: return default
        cur = cur[k]
    return cur

def _first_present(d, keys):
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    return None

def main():
    ap = argparse.ArgumentParser(description="Reproducible demo wrapper")
    ap.add_argument("image", help="Path to image file")
    ap.add_argument("symptom_text", nargs="?", default=None, help="Optional symptom text")
    args = ap.parse_args()

    raw = _run_router(args.image, args.symptom_text)

    # --- Normalize to requested schema
    out = {}

    # stage
    out["stage"] = raw.get("stage")

    # thresholds (binary)
    thr = _get(raw, ["binary", "threshold_used"], None)
    if thr is not None:
        out["thresholds"] = {"binary": float(thr)}

    # alpha
    alpha = _first_present(raw, ["alpha", "α"])
    if alpha is None:
        alpha = D["alpha"]
    out["alpha"] = float(alpha)
    out["α"] = float(alpha)

    # top-k image
    topk_image = _get(raw, ["multiclass", "topk_image"], None)
    if topk_image is not None:
        out["topk_image"] = topk_image

    # top-k text (present only if text given)
    topk_text = (
        _get(raw, ["text", "topk_text"], None)
        or _get(raw, ["multiclass", "topk_text"], None)
        or _get(raw, ["nlp", "topk_text"], None)
    )
    if args.symptom_text and topk_text is not None:
        out["topk_text"] = topk_text

    # top-k fused (present only if text given)
    topk_fused = (
        raw.get("topk_fused")
        or _get(raw, ["fusion", "topk_fused"], None)
        or _get(raw, ["multiclass", "topk_fused"], None)
    )
    if args.symptom_text and topk_fused is not None:
        out["topk_fused"] = topk_fused

    # advisory (only if present)
    advisory = raw.get("advisory")
    if advisory:
        out["advisory"] = advisory

    print(json.dumps(out, indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())
