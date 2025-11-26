# Skin Disease Classifier – Submission Bundle

This folder contains the minimal artifacts to reproduce a demo run and view core metrics.

## Contents

```
artifacts/
  ├─ checkpoints/
  │   ├─ binary/best.pt
  │   ├─ multiclass/best.pt
  │   └─ symptom_bert/            # (present only if you included it)
  ├─ calibration/multiclass_temp.json
  └─ reports/
      ├─ demo_router.json
      ├─ fusion_* (if present)
      └─ other training/eval reports you copied
labels_26.txt
label_map.json
data/samples/sample.jpg
```

## Environment setup (macOS / MPS or CPU)

```bash
python -V                          # Python 3.10+ recommended
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
# CPU wheels (portable). If you have Apple GPU, you can still run with --device mps.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install timm pytorch-lightning==2.3.3 torchmetrics scikit-learn matplotlib pandas pillow transformers

# (Optional for Apple Silicon acceleration)
echo 'export PYTORCH_ENABLE_MPS_FALLBACK=1' >> ~/.zshrc && source ~/.zshrc
```

## One-line demo (router)

Run from repository root, targeting files inside `deliverables/`:

```bash
python -m src.router \
  --image deliverables/data/samples/sample.jpg \
  --binary_ckpt deliverables/artifacts/checkpoints/binary/best.pt \
  --multiclass_ckpt deliverables/artifacts/checkpoints/multiclass/best.pt \
  --multiclass_temp deliverables/artifacts/calibration/multiclass_temp.json \
  --labels_txt deliverables/labels_26.txt \
  --alpha 0.8 \
  --out deliverables/artifacts/reports/demo_router.json \
  --device mps
```

> If you included `artifacts/checkpoints/symptom_bert/`, you may add:
> `--text_ckpt deliverables/artifacts/checkpoints/symptom_bert --symptom_text "itchy red patches on arms"`

## How to run individual parts

**Binary quick check**: The router JSON prints the binary stage with `threshold_used = 0.95`.

**Multiclass eval**: If your repo exposes the evaluator,
```bash
python -m src.stages.multiclass.evaluate \
  --ckpt deliverables/artifacts/checkpoints/multiclass/best.pt \
  --temp_file deliverables/artifacts/calibration/multiclass_temp.json
```
(If module import fails, skip—reports have already been copied.)

**Text inference** (if shipped):
```bash
python -m src.nlp.infer \
  --text "itchy red patches on arms, burning when scratched" \
  --ckpt deliverables/artifacts/checkpoints/symptom_bert
```

## Thresholds, temperature, and fusion α

- **Binary threshold**: 0.95 (reduces false “Unhealthy”).  
- **Temperature T (multiclass)**: ~1.7908 (improves calibration, ECE ~0.0265 on your run).  
- **Fusion α**: 0.8 (favor text), since text >> image on your current val pairs.

## Safety disclaimer

- Not a medical device. Not for diagnosis/treatment.
- Predictions may be wrong on out-of-distribution data.
- Always consult a qualified clinician; never use in emergencies.

## Reproducible Demo
Run with symptom text:

```bash
python router.py "data/multiclass/Val/Chickenpox/CHP_01_01_10_1.jpg" "itchy red bumps on face"
```

Run without symptom text:

```bash
python router.py "data/multiclass/Val/Chickenpox/CHP_01_01_10_1.jpg"
```


## Held-out Test Metrics (Oct 2, 2025)
Binary (test): see artifacts/reports/binary_test_report.txt (target acc ≥ 0.95)
Multiclass (test): see artifacts/reports/mc_test_report.txt and artifacts/reports/mc_test_confusion.png (targets: macro-F1 ≥ 0.65, top-1 acc ≥ 0.70)
Text (test): see artifacts/reports/text_test_report.txt (target macro-F1 ≥ 0.60)

## Held-out Test Metrics (Oct 2, 2025)
Binary (test): see artifacts/reports/binary_test_report.txt (target acc ≥ 0.95)
Multiclass (test): see artifacts/reports/mc_test_report.txt and artifacts/reports/mc_test_confusion.png (targets: macro-F1 ≥ 0.65, top-1 acc ≥ 0.70)
Text (test): see artifacts/reports/text_test_report.txt (target macro-F1 ≥ 0.60)

### Calibration sanity (val)
- T = 1.7908
- ECE (val) after T = 0.0169 (before = 0.1087)
- Checked on 2025-10-02 19:04:14
