import os, glob, json, sys
from collections import Counter, defaultdict

IMG_EXTS = {'.jpg','.jpeg','.png','.bmp','.webp'}

def count_images(root):
    return sum(1 for p in glob.iglob(os.path.join(root,'**','*'), recursive=True)
               if os.path.splitext(p)[1].lower() in IMG_EXTS)

def list_immediate_subdirs(p):
    return sorted([d for d in os.listdir(p) if os.path.isdir(os.path.join(p,d))])

def count_images_per_class(split_dir):
    out = {}
    if not os.path.exists(split_dir):
        raise FileNotFoundError(split_dir)
    for c in list_immediate_subdirs(split_dir):
        cdir = os.path.join(split_dir, c)
        out[c] = count_images(cdir)
    return out

# 1) Binary: check Healthy/Unhealthy per split
for split in ["train","val","test"]:
    split_dir = os.path.join("data","binary",split)
    subs = list_immediate_subdirs(split_dir)
    print(f"[binary/{split}] subfolders:", subs)
    need = {"Healthy","Unhealthy"}
    missing = need - set(subs)
    assert not missing, f"{split_dir} is missing: {sorted(missing)}"
    print(f"[binary/{split}] images total:", count_images(split_dir))

print("-"*60)

# 2) Multiclass: infer canonical 26 classes from *train*, then validate other splits
mc = {s: os.path.join("data","multiclass",s) for s in ["train","val","test"]}
train_classes = list_immediate_subdirs(mc["train"])
print(f"[multiclass/train] classes ({len(train_classes)}):", train_classes)
if len(train_classes) != 26:
    print("⚠️ Expected 26 classes in train but found", len(train_classes))
    # show what you actually have by counts
    counts = count_images_per_class(mc["train"])
    print("Counts per class (train):", json.dumps(counts, indent=2))
    sys.exit(1)

os.makedirs("artifacts", exist_ok=True)
with open("artifacts/labels_26.txt","w") as f:
    for c in train_classes: f.write(c+"\n")
print("Wrote artifacts/labels_26.txt")

# Validate val/test contain the same class set (order can differ)
for split in ["val","test"]:
    split_classes = set(list_immediate_subdirs(mc[split]))
    missing = set(train_classes) - split_classes
    extra   = split_classes - set(train_classes)
    if missing:
        print(f"⚠️ [multiclass/{split}] missing classes:", sorted(missing))
    if extra:
        print(f"⚠️ [multiclass/{split}] extra classes:", sorted(extra))
    # Counts per class
    counts = count_images_per_class(mc[split])
    print(f"[multiclass/{split}] images total:", sum(counts.values()))
    # Warn about empty classes
    empties = [c for c in train_classes if counts.get(c,0)==0]
    if empties:
        print(f"⚠️ [multiclass/{split}] empty classes:", empties)

# 3) Symptoms CSV existence
for split in ["train","val","test"]:
    p = os.path.join("data","symptoms",f"symptom_{split}.csv")
    assert os.path.exists(p), f"Missing {p}"
print("Symptoms CSVs present.")

print("\nAll checks done. ✅")
