# setup_labels.py
import json, os, sys

PATH_TXT_DIR = "artifacts/checkpoints/symptom_bert"
PATH_LABELS_MC = "labels_26.txt"
PATH_LABELS_BIN = "artifacts/labels_binary.txt"

def main():
    # multiclass (26)
    labels_json = os.path.join(PATH_TXT_DIR, "labels.json")
    if os.path.isfile(labels_json):
        labels = json.load(open(labels_json, "r")).get("labels", [])
        assert labels and len(labels) >= 2, "labels.json missing or invalid"
        with open(PATH_LABELS_MC, "w") as f:
            for l in labels:
                f.write(l + "\n")
        print(f"[OK] wrote {PATH_LABELS_MC} ({len(labels)} labels)")
    else:
        print(f"[WARN] {labels_json} not found. Skipping labels_26.txt.")

    # binary (2)
    os.makedirs(os.path.dirname(PATH_LABELS_BIN), exist_ok=True)
    with open(PATH_LABELS_BIN, "w") as f:
        f.write("Healthy\nUnhealthy\n")
    print(f"[OK] wrote {PATH_LABELS_BIN} (2 labels: Healthy, Unhealthy)")

if __name__ == "__main__":
    main()
