# make_label_map.py
# Build a mapping from "fine"/external labels in your CSVs to your 26 canonical classes.
# Outputs:
#   - label_map.json         (proposed mappings into the 26-class taxonomy)
#   - label_map_TODO.txt     (labels that still need manual mapping)

import os
import re
import json
import pandas as pd
from typing import List, Dict, Optional

# ---- CONFIG ----
CANON_PATH = "artifacts/labels_26.txt"
CSV_PATHS = [
    "/Users/sandunwithanage/Documents/Data001/Symptoms_Data/symptom_train.csv",
    "/Users/sandunwithanage/Documents/Data001/Symptoms_Data/symptom_val.csv",
    "/Users/sandunwithanage/Documents/Data001/Symptoms_Data/symptom_test.csv",
]
OUT_JSON = "label_map.json"
OUT_TODO = "label_map_TODO.txt"


def read_canonical(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        labels = [ln.strip() for ln in f if ln.strip()]
    return labels


def load_unique_labels(csvs: List[str]) -> List[str]:
    uniq = set()
    for p in csvs:
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        if "label" not in df.columns:
            continue
        uniq |= set(df["label"].dropna().astype(str).unique().tolist())
    return sorted(uniq)


def _contains(pattern: str, s: str) -> bool:
    return re.search(pattern, s) is not None


def propose_map(src_label: str, canon: List[str]) -> Optional[str]:
    """
    Heuristic mapping rules -> map ONLY into your known 26-class taxonomy.
    Returns canonical label string or None if unsure.
    """
    s = src_label.strip()
    sl = s.lower().replace(" ", "_")

    # if already canonical, no mapping required
    if s in canon:
        return None

    # short-hands for existence check
    def pick(name: str) -> Optional[str]:
        return name if name in canon else None

    # ---- STRICT / HIGH-CONFIDENCE RULES (alphabetical-ish by target) ----
    # Acne family
    if _contains(r"\bacne\b", sl) or _contains(r"\bfolliculitis\b", sl):
        cand = pick("Acne")
        if cand: return cand

    # Actinic keratosis
    if _contains(r"actinic.*keratos", sl):
        cand = pick("Actinic_Keratosis")
        if cand: return cand

    # Benign tumors (e.g., keloid)
    if _contains(r"\bkeloid\b", sl):
        cand = pick("Benign_tumors")
        if cand: return cand

    # Bullous disorders (rarely used as mapping; keep conservative)
    # (No strong automatic rule here to avoid bad mappings)

    # Candidiasis
    if _contains(r"candidiasis|candida|thrush", sl):
        cand = pick("Candidiasis")
        if cand: return cand

    # Chickenpox / Cowpox / HFMD / Measles / Monkeypox
    if _contains(r"\bchickenpox\b|varicella", sl):
        cand = pick("Chickenpox")
        if cand: return cand
    if _contains(r"\bcowpox\b", sl):
        cand = pick("Cowpox")
        if cand: return cand
    if _contains(r"hand.*foot.*mouth|hfmd", sl):
        cand = pick("HFMD")
        if cand: return cand
    if _contains(r"\bmeasles\b|rubeola", sl):
        cand = pick("Measles")
        if cand: return cand
    if _contains(r"\bmonkeypox\b|mpox", sl):
        cand = pick("Monkeypox")
        if cand: return cand

    # Drug eruptions / Urticaria / hives
    if _contains(r"\bursticaria\b|\bhive(s)?\b", sl):
        cand = pick("DrugEruption")
        if cand: return cand
    if _contains(r"drug.*eruption|adverse.*drug|drug.*rash", sl):
        cand = pick("DrugEruption")
        if cand: return cand

    # Eczema / Dermatitis bucket (broad inflammatory dermatoses)
    if _contains(r"eczema|dermatitis|cellulitis|impetigo|pityriasis_?rosea", sl):
        cand = pick("Eczema")
        if cand: return cand

    # Infestations & bites
    if _contains(r"\bscabies\b|mite|lice|pediculosis|bug|bite", sl):
        cand = pick("Infestations_Bites")
        if cand: return cand

    # Lichen
    if _contains(r"lichen(_planus)?", sl):
        cand = pick("Lichen")
        if cand: return cand

    # Lupus
    if _contains(r"\blupus\b", sl):
        cand = pick("Lupus")
        if cand: return cand

    # Moles / Nevi
    if _contains(r"\bmole(s)?\b|nevus|nevi", sl):
        cand = pick("Moles")
        if cand: return cand

    # Psoriasis
    if _contains(r"\bpsoriasis\b", sl):
        cand = pick("Psoriasis")
        if cand: return cand

    # Rosacea
    if _contains(r"\brosacea\b", sl):
        cand = pick("Rosacea")
        if cand: return cand

    # Seborrheic dermatitis/keratosis -> Seborrh_Keratoses
    if _contains(r"seborrhe(ic|ic)_?dermatitis|seborrhe(ic|ic)_?keratos", sl):
        cand = pick("Seborrh_Keratoses")
        if cand: return cand
    if _contains(r"keratosis", sl) and pick("Seborrh_Keratoses"):
        # generic keratosis (fallback to your only keratosis-like class)
        return "Seborrh_Keratoses"

    # Skin cancer (all types -> SkinCancer)
    if _contains(r"skin.*cancer|basal.*cell.*carcinoma|squamous.*cell.*carcinoma|melanoma|bcc|scc", sl):
        cand = pick("SkinCancer")
        if cand: return cand

    # Sun damage / Melasma / Photo ageing
    if _contains(r"\bmelasma\b|photo(age|aging)|sun(_| )damage|solar", sl):
        cand = pick("Sun_Sunlight_Damage")
        if cand: return cand

    # Tinea / ringworm / dermatophyte
    if _contains(r"tinea|ringworm|dermatophyt", sl):
        cand = pick("Tinea")
        if cand: return cand

    # Vascular tumors / hemangiomas / vascular lesions
    if _contains(r"vascular.*(lesion|tumou?r)|hemangioma|angioma|capillary|port.?wine", sl):
        cand = pick("Vascular_Tumors")
        if cand: return cand

    # Vasculitis
    if _contains(r"\bvasculitis\b", sl):
        cand = pick("Vasculitis")
        if cand: return cand

    # Vitiligo
    if _contains(r"\bvitiligo\b", sl):
        cand = pick("Vitiligo")
        if cand: return cand

    # Viral papules: warts / molluscum -> Warts (closest bucket you have)
    if _contains(r"\bwart(s)?\b|molluscum", sl):
        cand = pick("Warts")
        if cand: return cand

    # ---- SUBSTRING / TOKEN OVERLAP FALLBACK ----
    # As a very last resort, try to match tokens into canonical names.
    toks = [t for t in re.split(r"[^a-z0-9]+", sl) if t]
    for c in canon:
        cl = c.lower()
        if any(t and t in cl for t in toks):
            return c

    # No confident mapping
    return None


def main():
    if not os.path.exists(CANON_PATH):
        raise SystemExit(f"❌ Missing canonical file: {CANON_PATH}")

    canon = read_canonical(CANON_PATH)
    canon_set = set(canon)
    print("Canonical classes (26):", canon)

    all_csv_labels = load_unique_labels(CSV_PATHS)
    unknown = [lab for lab in all_csv_labels if lab not in canon_set]

    print(f"\nFound {len(unknown)} non-canonical labels to map.\n")

    proposed: Dict[str, str] = {}
    for lab in unknown:
        dest = propose_map(lab, canon)
        if dest is not None:
            proposed[lab] = dest

    # Seed fixes / explicit overrides (edit if you need to force)
    seed_overrides = {
        # From our earlier review:
        "Molluscum": "Warts",
        "Cellulitis": "Eczema",
        "Folliculitis": "Acne",
        "Impetigo": "Eczema",
        "Keloid": "Benign_tumors",
        "Melasma": "Sun_Sunlight_Damage",
        "Pityriasis_Rosea": "Eczema",
        "Scabies": "Infestations_Bites",
        "Urticaria": "DrugEruption",
        # Keep only canonical targets from your 26-class list.
    }

    # Merge: explicit overrides take precedence over heuristics
    merged = {**proposed, **seed_overrides}

    # Keep only mappings that point to canonical names
    cleaned = {k: v for k, v in merged.items() if v in canon_set}

    # Write JSON
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)
    print(f"💾 Wrote {OUT_JSON} with {len(cleaned)} mappings.")

    # TODO list
    still_unmapped = [lab for lab in unknown if lab not in cleaned]
    if still_unmapped:
        with open(OUT_TODO, "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(still_unmapped)))
        print(f"⚠️  {len(still_unmapped)} labels still need manual mapping → {OUT_TODO}")
    else:
        print("✅ All non-canonical labels received a proposed mapping.")

    # Small preview
    if cleaned:
        print("\nExamples:")
        for i, (k, v) in enumerate(sorted(cleaned.items())[:10], 1):
            print(f"  {i:02d}. {k}  →  {v}")


if __name__ == "__main__":
    main()
