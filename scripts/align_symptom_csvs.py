import os, json, glob, re
import pandas as pd

LABELS_TXT   = "artifacts/labels_26.txt"
IN_DIR       = "data/symptoms"
OUT_DIR      = "data/symptoms_aligned"
MAP_DIR      = "artifacts/calibration"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MAP_DIR, exist_ok=True)

def norm(s): 
    return re.sub(r'[\s_\-]+','',str(s)).strip().lower()

# --- load 26-name canon
canon = [ln.strip() for ln in open(LABELS_TXT, encoding="utf-8") if ln.strip()]
canon_norm = {norm(x): x for x in canon}
canon_set  = set(canon)

# --- collect all labels seen in CSVs
csvs = sorted(glob.glob(f"{IN_DIR}/symptom_*.csv"))
if not csvs:
    print(f"[WARN] no CSVs matching {IN_DIR}/symptom_*.csv")
    raise SystemExit(0)

def find_label_col(df):
    for c in ["label","labels","diagnosis","class","target","y"]:
        if c in df.columns and df[c].dtype == object:
            return c
    # fallback: first object column
    for c in df.columns:
        if df[c].dtype == object:
            return c
    return None

summary = []
for csv in csvs:
    df = pd.read_csv(csv)
    col = find_label_col(df)
    if not col:
        print(f"[FAIL] cannot find label column in {csv}")
        continue

    # normalize labels to canon by case/spacing/underscore
    orig_unique = sorted(str(x) for x in df[col].dropna().unique())
    remap = {}
    unmapped = set()
    for u in orig_unique:
        nu = norm(u)
        if nu in canon_norm:
            remap[u] = canon_norm[nu]
        else:
            unmapped.add(u)

    # apply remap
    df[col] = df[col].map(lambda x: remap.get(str(x), str(x)))

    # Option A: drop rows still not in canon
    mask_in = df[col].isin(canon_set)
    kept = int(mask_in.sum())
    dropped = int((~mask_in).sum())
    out_csv = os.path.join(OUT_DIR, os.path.basename(csv))
    df[mask_in].to_csv(out_csv, index=False)

    # write a manual mapping template for later (Option B)
    if unmapped:
        template = {
            "NOTE": "Fill values with one of the 26 canonical labels from artifacts/labels_26.txt, then rerun this script to map instead of dropping.",
            "unmapped_to_target": {u: "" for u in sorted(unmapped)},
            "canon_labels": canon
        }
        map_path = os.path.join(MAP_DIR, f"symptom_manual_map_{os.path.basename(csv)}.json")
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(template, f, ensure_ascii=False, indent=2)

    summary.append({
        "csv": csv,
        "label_col": col,
        "classes_seen": len(orig_unique),
        "kept_rows": kept,
        "dropped_rows": dropped,
        "manual_map_json": os.path.join(MAP_DIR, f"symptom_manual_map_{os.path.basename(csv)}.json") if unmapped else None,
        "output": out_csv
    })

print("\n=== ALIGNMENT SUMMARY (Option A: dropped out-of-canon rows) ===")
for s in summary:
    name = os.path.basename(s["csv"])
    print(f"{name}: kept={s['kept_rows']} dropped={s['dropped_rows']} → {s['output']}")
    if s["manual_map_json"]:
        print(f"  manual map template: {s['manual_map_json']}")
print("\n[INFO] Use the aligned files in data/symptoms_aligned for your text model / router.")
