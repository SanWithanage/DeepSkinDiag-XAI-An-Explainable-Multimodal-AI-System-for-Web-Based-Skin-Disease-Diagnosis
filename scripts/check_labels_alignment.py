import os, json, re, glob, sys
import pandas as pd

# ---- paths (adjust if your repo differs)
LABELS_TXT   = "artifacts/labels_26.txt"
TRAIN_DIR    = "data/multiclass/train"
SYMPTOM_GLOB = "data/symptoms_aligned/symptom_*.csv"   # symptom_{train,val,test}.csv
TEXT_CKPT    = "artifacts/checkpoints/symptom_bert_hf"  # folder that should contain labels.json or config.json
OUT_DIR      = "artifacts/calibration"
os.makedirs(OUT_DIR, exist_ok=True)

def norm(s):
    return re.sub(r'[\s_\-]+', '', str(s)).strip().lower()

def read_labels_txt(p):
    with open(p, 'r', encoding='utf-8') as f:
        labels = [ln.strip() for ln in f if ln.strip()]
    return labels

def check_labels_vs_folders(labels):
    if not os.path.isdir(TRAIN_DIR):
        print(f"[FAIL] Train dir missing: {TRAIN_DIR}")
        return False
    folders = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR,d))])
    ok = set(labels) == set(folders)
    if ok and labels == folders:
        print("[PASS] labels_26.txt exactly matches training folder names and order.")
    elif ok:
        print("[WARN] Same label SET as folders, but ORDER differs.")
        tmp = os.path.join(OUT_DIR, "labels_26_from_folders.txt")
        with open(tmp, "w", encoding="utf-8") as f: f.write("\n".join(folders)+"\n")
        print(f"       Wrote folder order → {tmp}")
    else:
        missing_in_txt  = sorted(set(folders) - set(labels))
        extra_in_txt    = sorted(set(labels)  - set(folders))
        print("[FAIL] labels_26.txt set != folder set.")
        if missing_in_txt: print("  Missing in labels_26.txt:", missing_in_txt)
        if extra_in_txt:   print("  Extra in labels_26.txt  :", extra_in_txt)
        tmp = os.path.join(OUT_DIR, "labels_26_from_folders.txt")
        with open(tmp, "w", encoding="utf-8") as f: f.write("\n".join(folders)+"\n")
        print(f"  → Suggested file with folder names written to {tmp}")
    return ok

def find_label_col(df):
    # prefer common names; fallback: shortest object column with ~26 uniques
    candidates = [c for c in ["label","labels","diagnosis","class","target","y"] if c in df.columns]
    for c in candidates:
        if df[c].dtype == object: return c
    # fallback heuristic
    obj_cols = [c for c in df.columns if df[c].dtype == object]
    best = None; best_gap = 1e9
    for c in obj_cols:
        u = df[c].nunique(dropna=True)
        gap = abs(u - 26)
        if gap < best_gap:
            best, best_gap = c, gap
    return best

def check_symptom_csvs(labels):
    print("\n[STEP] Checking symptom CSV label set...")
    canon_by_norm = {norm(x): x for x in labels}
    ok_all = True
    for csv in sorted(glob.glob(SYMPTOM_GLOB)):
        try:
            df = pd.read_csv(csv)
        except Exception as e:
            print(f"[FAIL] Could not read {csv}: {e}")
            ok_all = False; continue
        col = find_label_col(df)
        if not col:
            print(f"[FAIL] No obvious label column in {csv}.")
            ok_all = False; continue
        uniq = sorted(str(x) for x in df[col].dropna().unique())
        # build mapping by normalization
        mapping = {}
        unknown = []
        for u in uniq:
            nu = norm(u)
            if nu in canon_by_norm:
                mapping[u] = canon_by_norm[nu]
            else:
                unknown.append(u)
        if unknown:
            print(f"[FAIL] {csv}: found labels not in the 26-name canon → {unknown}")
            ok_all = False
            continue
        src_set = set(mapping.keys())
        tgt_set = set(mapping.values())
        subset_ok = set(tgt_set).issubset(set(labels))
        exact_ok = subset_ok and all(k==v for k,v in mapping.items())
        if exact_ok:
            print(f"[PASS] {csv}: labels exactly match the 26 names (case/spacing). Column={col}")
        elif subset_ok:
            print(f"[WARN] {csv}: same SET but case/spacing differs. Column={col}")
            j = os.path.join(OUT_DIR, f"label_mapping_{os.path.basename(csv)}.json")
            with open(j,"w",encoding="utf-8") as f: json.dump(mapping, f, ensure_ascii=False, indent=2)
            print(f"       Wrote mapping → {j} (use it to normalize if you want strict match)")
        else:
            print(f"[FAIL] {csv}: label SET does not equal the 26-name set.")
            extra = sorted(tgt_set - set(labels))
            miss  = sorted(set(labels) - tgt_set)
            if extra: print("  Extra after mapping:", extra)
            if miss:  print("  Missing after mapping:", miss)
            ok_all = False
    return ok_all

def read_text_label_order():
    # try labels.json then config.json
    lj = os.path.join(TEXT_CKPT, "labels.json")
    cj = os.path.join(TEXT_CKPT, "config.json")
    if os.path.isfile(lj):
        j = json.load(open(lj,"r",encoding="utf-8"))
    elif os.path.isfile(cj):
        j = json.load(open(cj,"r",encoding="utf-8"))
    else:
        return None
    if isinstance(j, list):
        return j
    if isinstance(j, dict):
        if "id2label" in j and isinstance(j["id2label"], dict):
            id2label = j["id2label"]
            try:
                # HuggingFace keys often "0","1",...
                return [id2label[str(i)] for i in range(len(id2label))]
            except:
                # try int keys
                keys = sorted(id2label, key=lambda x: int(x))
                return [id2label[k] for k in keys]
        if "label2id" in j and isinstance(j["label2id"], dict):
            l2i = j["label2id"]
            inv = {v:k for k,v in l2i.items()}
            keys = [inv[i] for i in range(len(inv))]
            return keys
    return None

def check_text_order(labels):
    print("\n[STEP] Checking text model label ORDER...")
    order = read_text_label_order()
    if order is None:
        print(f"[WARN] No labels.json/config.json found in {TEXT_CKPT}. Skipping text-order check.")
        return True
    # basic set check
    if set(order) != set(labels):
        print("[FAIL] Text model label SET != labels_26.txt set.")
        print("  Missing in text:", sorted(set(labels) - set(order)))
        print("  Extra in text  :", sorted(set(order)  - set(labels)))
        return False
    if order == labels:
        print("[PASS] Text labels order matches labels_26.txt exactly.")
        # Write identity map for clarity
        idx_map = list(range(len(labels)))
    else:
        print("[WARN] Text labels order differs from labels_26.txt (this is OK if you reindex before fusion).")
        # Build index_map: for each target label (labels[i]), where is it in text-order?
        idx_map = [order.index(lbl) for lbl in labels]
        out = {
            "source": "text_labels",
            "order_in_text": order,
            "order_target": labels,
            "index_map": idx_map
        }
        j = os.path.join(OUT_DIR, "text_reindex.json")
        with open(j,"w",encoding="utf-8") as f: json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"       Wrote reindex map → {j}")
    return True

def main():
    ok_all = True
    if not os.path.isfile(LABELS_TXT):
        print(f"[FAIL] Missing {LABELS_TXT}. Create it with one label per line (26 lines).")
        sys.exit(2)
    labels = read_labels_txt(LABELS_TXT)
    if len(labels) != 26:
        print(f"[FAIL] {LABELS_TXT} should contain 26 labels, found {len(labels)}.")
        ok_all = False
    if not check_labels_vs_folders(labels): ok_all = False
    if not check_symptom_csvs(labels):      ok_all = False
    if not check_text_order(labels):        ok_all = False

    print("\n=== SUMMARY ===")
    if ok_all:
        print("[OK] Alignment checks completed with no blocking failures.")
    else:
        print("[ATTENTION] Fix the FAIL items above. Use the written artifacts in artifacts/calibration/ to help.")
    sys.exit(0 if ok_all else 1)

if __name__ == "__main__":
    main()
