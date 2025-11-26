# app.py — Sleek one-page UI (updated safely)
# - Robust, shape-safe loader (unchanged core)
# - Healthy override: show only "Healthy" reassurance (no suggestions)
# - Optional: "Show other possibilities (Top-3)" to reveal guidance (Healthy case)
# - Optional: low-confidence gate + "Show guidance anyway" (Uncertain case)
# - Dual panels (Western + Sinhala Ayurvedic — English) with Top-1/2/3 radio
# - Loads artifacts/guidance/guidance_en.json (safe fallback)
# - High-contrast cards, friendlier layout
# - Saves each uploaded image to artifacts/debug/last_input.jpg for debugging

import os, io, json, socket, datetime as dt
from typing import Dict, List, Tuple, Optional, Set

import torch
import torch.nn.functional as F
from PIL import Image
import timm
from torchvision import transforms, models as tvm
import gradio as gr
import pandas as pd
from pathlib import Path

# =========================
# Config
# =========================
TITLE = "SkinAI — Image Diagnosis (with Healthy Override)"
SUBTITLE = "Upload a skin photo. The app predicts 'Healthy' when the binary model is confident."
TOPK = 3
IMAGE_SIZE = 224
HEALTHY_MIN_CONFIDENCE_DEFAULT = 0.55
PRIORITIZE_BINARY_HEALTHY_DEFAULT = True
SWAP_BINARY_DEFAULT = False  # set True if your binary model index order is [Unhealthy, Healthy]

# Confidence gate (UI only; does NOT change predictions)
HIGH_CONF_GATE = 0.60  # hide disease guidance below this; show "Show guidance anyway" button

# Paths
PATH_BIN_CKPT   = "artifacts/checkpoints/binary/best.pt"
PATH_MC_CKPT    = "artifacts/checkpoints/multiclass/best.pt"
PATH_CALIB      = "artifacts/calibration/multiclass_temp.json"
PATH_LABELS_BIN = "artifacts/labels_binary.txt"
PATH_LABELS_26_CANDIDATES = ["labels_26.txt", "artifacts/labels_26.txt"]

# Where to save CSV/JSON exports (NEW)
OUT_DIR = Path("artifacts/reports/runs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# External guidance (you already have this)
PATH_GUIDANCE_JSON = "artifacts/guidance/guidance_en.json"

# Optional text model (unused in this build)
TEXT_DIR  = "artifacts/checkpoints/symptom_bert"
TEXT_FILES = ["config.json","tokenizer.json","labels.json","pytorch_model.bin"]

STYLE_WESTERN = "Western (Dermatology) — English"
STYLE_AYUR    = "Sinhala Ayurvedic (Traditional) — English"

# =========================
# Small utils
# =========================
def _device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")

def _read_lines(p: str) -> List[str]:
    with open(p, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]

def _first_existing(path_list: List[str]) -> Optional[str]:
    for p in path_list:
        if os.path.isfile(p):
            return p
    return None

def _load_temperature(p: str) -> float:
    if not os.path.isfile(p): return 1.0
    try:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        return float(d.get("T", 1.0))
    except Exception:
        return 1.0

def _softmax(x: torch.Tensor, T: float = 1.0) -> torch.Tensor:
    return F.softmax(x / T, dim=-1)

def _time_tag() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")

def _pick_free_port(pref: int = 7860, max_tries: int = 40) -> int:
    def _is_free(port: int) -> bool:
        import socket as _s
        with _s.socket(_s.AF_INET, _s.SOCK_STREAM) as s:
            s.settimeout(0.1)
            return s.connect_ex(("127.0.0.1", port)) != 0
    if _is_free(pref): return pref
    for d in range(1, max_tries+1):
        if _is_free(pref+d): return pref+d
    return 0

def _bar_html(rows: List[Tuple[str, float]], title: str) -> str:
    def pretty(s: str) -> str: return s.replace("_", " ")
    parts = [f"<div style='font-weight:600;margin:6px 0 2px'>{title}</div>"]
    for label, p in rows:
        pct = int(round(p * 100))
        parts.append(
            f"""
            <div style="display:flex;align-items:center; gap:8px; margin:4px 0;">
              <div style="width:200px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis">{pretty(label)}</div>
              <div style="flex:1; background:#eee; height:12px; border-radius:6px; overflow:hidden;">
                <div style="width:{pct}%; height:100%; background:#4f46e5;"></div>
              </div>
              <div style="width:52px; text-align:right; font-variant-numeric:tabular-nums">{pct}%</div>
            </div>
            """
        )
    return "<div style='font-family:Inter,system-ui,Segoe UI,Arial,sans-serif;'>" + "\n".join(parts) + "</div>"

def _variants_of_state_dict(sd: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
    keys = list(sd.keys())
    variants = [sd]
    for prefix in ("module.", "model.", "net."):
        if all(k.startswith(prefix) for k in keys):
            variants.append({k[len(prefix):]: v for k, v in sd.items()})
    uniq, seen = [], set()
    for d in variants:
        marker = (len(d),) + tuple(sorted(list(d.keys())[:8]))
        if marker not in seen:
            uniq.append(d); seen.add(marker)
    return uniq

def _shape_filtered(sd: Dict[str, torch.Tensor], model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    m_sd = model.state_dict()
    out = {}
    for k, v in sd.items():
        if k in m_sd and hasattr(v, "shape") and hasattr(m_sd[k], "shape") and tuple(v.shape) == tuple(m_sd[k].shape):
            out[k] = v
    return out

def _extract_labelnames_from_ckpt(d: dict) -> Optional[List[str]]:
    for k in ("labels", "class_names", "classes", "idx_to_class"):
        if k in d:
            v = d[k]
            if isinstance(v, dict):
                try:
                    v = [v[i] for i in sorted(v, key=lambda x: int(x))]
                except Exception:
                    v = list(v.values())
            if isinstance(v, (list, tuple)) and all(isinstance(x, (str, int)) for x in v):
                return [str(x) for x in v]
    return None

# =========================
# Load labels + temperature
# =========================
bin_labels_file = _read_lines(PATH_LABELS_BIN) if os.path.isfile(PATH_LABELS_BIN) else ["Healthy","Unhealthy"]
assert "Healthy" in bin_labels_file and "Unhealthy" in bin_labels_file, \
    f"{PATH_LABELS_BIN} must contain 'Healthy' and 'Unhealthy' (one per line)."

TEMPERATURE = _load_temperature(PATH_CALIB)

labels_26_path = _first_existing(PATH_LABELS_26_CANDIDATES)
if labels_26_path:
    mc_labels_file = _read_lines(labels_26_path)
else:
    mc_labels_file = None  # will try from ckpt

# =========================
# Preprocessing
# =========================
img_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

# =========================
# Device
# =========================
device = _device()
print(f"[INFO] Using device: {device.type}")

# =========================
# Binary model (timm)
# =========================
bin_model = timm.create_model("tf_efficientnet_b0", pretrained=False, num_classes=2)
bin_state = torch.load(PATH_BIN_CKPT, map_location="cpu")
if isinstance(bin_state, dict) and "state_dict" in bin_state:
    bin_state = bin_state["state_dict"]

loaded_bin = False
if isinstance(bin_state, dict):
    for sdv in _variants_of_state_dict(bin_state):
        try:
            filtered = _shape_filtered(sdv, bin_model)
            missing, unexpected = bin_model.load_state_dict(filtered, strict=False)
            print(f"[OK] Loaded binary weights with {len(filtered)} matched keys "
                  f"({len(missing)} missing / {len(unexpected)} unexpected after filter).")
            loaded_bin = True
            break
        except Exception:
            pass
if not loaded_bin:
    print("[WARN] Could not match any binary checkpoint keys by shape. Running with randomly initialized binary head.")
bin_model.eval().to(device)

# =========================
# Multiclass model (auto-detect + shape-safe)
# =========================
mc_ckpt = torch.load(PATH_MC_CKPT, map_location="cpu")
ckpt_labels = None
arch_hint = None
raw_state = None

if isinstance(mc_ckpt, dict):
    ckpt_labels = _extract_labelnames_from_ckpt(mc_ckpt)
    if "arch" in mc_ckpt and isinstance(mc_ckpt["arch"], str):
        arch_hint = mc_ckpt["arch"]
    if "state_dict" in mc_ckpt and isinstance(mc_ckpt["state_dict"], dict):
        raw_state = mc_ckpt["state_dict"]
if raw_state is None:
    raw_state = mc_ckpt if isinstance(mc_ckpt, dict) else mc_ckpt

if ckpt_labels and len(ckpt_labels) >= 2:
    mc_labels = [str(x) for x in ckpt_labels]
    print(f"[OK] Using {len(mc_labels)} labels from checkpoint.")
else:
    if mc_labels_file:
        mc_labels = mc_labels_file
        print(f"[OK] Using {len(mc_labels)} labels from {labels_26_path}.")
    else:
        raise RuntimeError("Could not fetch multiclass labels: neither checkpoint labels nor labels_26.txt present.")

def _build_candidates(num_classes: int, hint: Optional[str]):
    cands = []
    if hint:
        try:
            cands.append((timm.create_model(hint, pretrained=False, num_classes=num_classes), f"timm:{hint}"))
        except Exception:
            pass
        if hint == "efficientnet_b0":
            m = tvm.efficientnet_b0(weights=None)
            in_feat = m.classifier[1].in_features
            m.classifier[1] = torch.nn.Linear(in_feat, num_classes)
            cands.append((m, "torchvision:efficientnet_b0"))
        if hint == "resnet50":
            m = tvm.resnet50(weights=None)
            in_feat = m.fc.in_features
            m.fc = torch.nn.Linear(in_feat, num_classes)
            cands.append((m, "torchvision:resnet50"))
    for a in ["tf_efficientnet_b0", "efficientnet_b0", "mobilenetv3_large_100", "resnet34", "convnext_tiny"]:
        try:
            cands.append((timm.create_model(a, pretrained=False, num_classes=num_classes), f"timm:{a}"))
        except Exception:
            pass
    try:
        m = tvm.efficientnet_b0(weights=None); in_feat = m.classifier[1].in_features
        m.classifier[1] = torch.nn.Linear(in_feat, num_classes)
        cands.append((m, "torchvision:efficientnet_b0"))
    except Exception:
        pass
    try:
        m = tvm.resnet50(weights=None); in_feat = m.fc.in_features
        m.fc = torch.nn.Linear(in_feat, num_classes)
        cands.append((m, "torchvision:resnet50"))
    except Exception:
        pass
    return cands

def _best_model_for_ckpt(num_classes: int, arch_hint: Optional[str], raw_state: Dict[str, torch.Tensor]):
    sd_variants = _variants_of_state_dict(raw_state) if isinstance(raw_state, dict) else []
    if not sd_variants:
        raise RuntimeError("Checkpoint is not a dict (state_dict).")

    candidates = _build_candidates(num_classes, arch_hint)
    if not candidates:
        raise RuntimeError("No candidate backbones could be constructed.")

    best = None
    for model, tag in candidates:
        m_sd = model.state_dict()
        for v_idx, sdv in enumerate(sd_variants):
            filt = {k: v for k, v in sdv.items() if k in m_sd and tuple(v.shape) == tuple(m_sd[k].shape)}
            match = len(filt)
            if best is None or match > best["match"]:
                best = dict(model=model, tag=tag, sd_variant=v_idx, matched_keys=filt, match=match, total=len(sdv))
    if best is None or best["match"] == 0:
        raise RuntimeError("Could not match any parameters by shape for the multiclass checkpoint on any backbone.")

    missing, unexpected = best["model"].load_state_dict(best["matched_keys"], strict=False)
    print(f"[OK] Multiclass backbone chosen: {best['tag']} (sd variant #{best['sd_variant']}). "
          f"Matched {best['match']} keys / {best['total']} in ckpt "
          f"({len(missing)} missing / {len(unexpected)} unexpected after filtered load).")
    return best["model"], best

mc_model, _best = _best_model_for_ckpt(num_classes=len(mc_labels), arch_hint=arch_hint, raw_state=raw_state)
mc_model.eval().to(device)

# Optional text model presence
ENABLE_TEXT = all(os.path.isfile(os.path.join(TEXT_DIR, f)) for f in TEXT_FILES)
if not ENABLE_TEXT:
    print("[WARN] Text model file missing: pytorch_model.bin. Running without symptom fusion.")

# =========================
# Guidance library (Western & Sinhala Ayurvedic — English)
# =========================
LOW_CONF = 0.55
CANCERY: Set[str] = {
    "skincancer","melanoma","basalcellcarcinoma","squamouscellcarcinoma","bcc","scc","cancer"
}

# Built-in generic fallbacks (only used if JSON missing)
GENERIC_WESTERN = [
    "Photograph the area in daylight every 2–3 days to track changes.",
    "Use a gentle, fragrance-free moisturizer; avoid harsh scrubs and peels.",
    "Use broad-spectrum sunscreen on exposed areas during the day.",
    "Avoid new cosmetics for 72 hours and patch-test any new product.",
]
GENERIC_AYUR = [
    "Keep the area clean and dry; avoid friction and tight clothing.",
    "Use a cool compress (clean cloth, cool boiled water) for comfort.",
    "For any new natural preparation, patch-test first (small area, 24h). Stop if irritation appears.",
    "If symptoms worsen or persist, consult a dermatologist or an Ayurvedic practitioner.",
]

# Load external guidance file if present
GUIDE = None
if os.path.isfile(PATH_GUIDANCE_JSON):
    try:
        with open(PATH_GUIDANCE_JSON, "r", encoding="utf-8") as f:
            GUIDE = json.load(f)
        assert isinstance(GUIDE, dict) and "western" in GUIDE and "ayur" in GUIDE, "Invalid guidance JSON structure."
        assert "generic" in GUIDE["western"] and "generic" in GUIDE["ayur"], "Guidance JSON missing 'generic' keys."
        print(f"[GUIDE] Loaded: {PATH_GUIDANCE_JSON}")
    except Exception as e:
        print(f"[GUIDE] Failed to load {PATH_GUIDANCE_JSON} ({e}). Falling back to built-ins.")
        GUIDE = None
else:
    print(f"[GUIDE] Not found: {PATH_GUIDANCE_JSON}. Using built-in generic tips.")

def get_tips(style: str, top_label: str) -> List[str]:
    use_western = style.startswith("Western")
    if GUIDE:
        bucket = GUIDE["western"] if use_western else GUIDE["ayur"]
        return bucket.get(top_label, bucket["generic"])
    else:
        return GENERIC_WESTERN if use_western else GENERIC_AYUR

def triage_level(top_label: str, top_conf: float, risk_flags: Set[str]) -> str:
    if (top_label.lower().replace(" ","").replace("_","") in CANCERY) or \
       ("cancer_like" in risk_flags) or ("rapid_change" in risk_flags):
        return "red"
    if top_conf < LOW_CONF or ("system_uncertain" in risk_flags):
        return "amber"
    return "green"

def build_suggestions(top_label: str, top_conf: float, risk_flags: Set[str],
                      healthy_prob: float, unhealthy_prob: float, healthy_threshold: float,
                      style: str) -> Dict:
    triage = triage_level(top_label, top_conf, risk_flags)

    if triage == "red":
        banner, detail = "⚠️ Urgent attention recommended", (
            "Some features look concerning. Please seek a clinician’s advice soon—"
            "especially if the spot changes shape, color, bleeds, or you feel unwell."
        )
    elif triage == "amber":
        banner, detail = "⚠️ Caution advised", (
            "The model is not fully certain. Monitor closely, reduce irritants, and consider a dermatology visit "
            "if it persists beyond 1–2 weeks or worsens."
        )
    else:
        banner, detail = "✅ Looks mild", "General skin-care steps may help. Keep monitoring for any changes."

    if triage == "red":
        next_steps = [
            "Book a dermatology appointment as soon as possible.",
            "Bring clear photos taken in daylight over several days.",
            "Note symptom timeline, triggers, and any pain, fever, or bleeding.",
        ]
    elif triage == "amber":
        next_steps = [
            "If no improvement in 1–2 weeks, or if it spreads rapidly, see a clinician.",
            "Minimize friction/irritants; keep the area clean and moisturized.",
        ]
    else:
        next_steps = [
            "Follow gentle care for 7–10 days and reassess.",
            "If new symptoms appear (severe pain, fever, bleeding), escalate to a clinician.",
        ]

    tips = get_tips(style, top_label)
    conf_line = (f"Model confidence: {top_conf:.2f} (top class: {top_label}) · "
                 f"Healthy {healthy_prob:.2f} vs Unhealthy {unhealthy_prob:.2f} (thr {healthy_threshold:.2f})")
    disclaimer = ("**This app provides guidance only and is not a medical diagnosis.** "
                  "Always consult a qualified professional for medical concerns.")

    return dict(
        triage=triage, banner=banner, banner_detail=detail,
        confidence=conf_line, next_steps=next_steps, tips=tips, disclaimer=disclaimer
    )

def render_suggestions_html(sug: Dict) -> str:
    palette = {
        "red":   {"bg":"#fef2f2", "bd":"#ef4444", "tx":"#0b1220", "icon":"❗"},
        "amber": {"bg":"#fff7ed", "bd":"#f59e0b", "tx":"#0b1220", "icon":"⚠️"},
        "green": {"bg":"#f0fdf4", "bd":"#22c55e", "tx":"#0b1220", "icon":"✅"},
    }
    p = palette.get(sug["triage"], palette["amber"])
    return f"""
    <div style="
        display:block;
        border:1px solid {p['bd']};
        background:{p['bg']};
        border-radius:12px;
        margin-top:10px;
        font-family:Inter,system-ui,Segoe UI,Arial,sans-serif;
        color:{p['tx']};
        line-height:1.5;
        font-size:16px;">
      <div style="display:flex;gap:10px;align-items:flex-start;">
        <div style="width:6px;background:{p['bd']};border-top-left-radius:12px;border-bottom-left-radius:12px;"></div>
        <div style="padding:14px 14px 14px 6px;flex:1;">
          <div style="font-weight:800;margin-bottom:6px;">{p['icon']} {sug["banner"]}</div>
          <div style="margin-bottom:12px;">{sug["banner_detail"]}</div>
          <div style="margin-bottom:12px;"><strong>Confidence:</strong> {sug["confidence"]}</div>
          <div style="font-weight:700;margin-bottom:6px;">What to do next</div>
          <ul style="margin:0 0 12px 20px;padding:0;list-style:disc;">
            {''.join(f'<li style="margin:4px 0;">{x}</li>' for x in sug["next_steps"])}
          </ul>
          <div style="font-weight:700;margin-bottom:6px;">Care tips</div>
          <ul style="margin:0 0 12px 20px;padding:0;list-style:disc;">
            {''.join(f'<li style="margin:4px 0;">{x}</li>' for x in sug["tips"])}
          </ul>
          <div style="margin-top:8px;font-size:0.9rem;opacity:0.9;">
            <em>{sug["disclaimer"]}</em>
          </div>
        </div>
      </div>
    </div>
    """

# =========================
# Inference
# =========================
@torch.inference_mode()
def predict_image(
    img: Image.Image,
    swap_binary_order: bool,
    healthy_threshold: float,
    prioritize_healthy: bool
) -> Dict:

    x = img_tf(img.convert("RGB")).unsqueeze(0).to(device)

    # --- Binary head ---
    bin_logits = bin_model(x)
    bin_probs_raw = _softmax(bin_logits, T=1.0).squeeze(0).cpu()  # [2]

    file_order = list(bin_labels_file)            # e.g. ["Healthy","Unhealthy"]
    file_order_swapped = [file_order[1], file_order[0]]
    chosen_order = file_order_swapped if swap_binary_order else file_order

    bin_out = {chosen_order[i]: float(bin_probs_raw[i]) for i in range(2)}
    healthy_prob = bin_out.get("Healthy", 0.0)
    unhealthy_prob = bin_out.get("Unhealthy", 0.0)
    bin_pairs = sorted(bin_out.items(), key=lambda z: z[1], reverse=True)[:TOPK]

    # --- Multiclass head ---
    mc_logits = mc_model(x)
    mc_probs = _softmax(mc_logits, T=TEMPERATURE).squeeze(0).cpu()
    mc_dict = {mc_labels[i]: float(mc_probs[i]) for i in range(len(mc_labels))}
    mc_top = sorted(mc_dict.items(), key=lambda z: z[1], reverse=True)[:TOPK]

    # Fused == image-only
    fused_top = mc_top[:]

    # --- Final decision with Healthy override ---
    fused_label, fused_prob = fused_top[0]
    if prioritize_healthy and healthy_prob >= unhealthy_prob and healthy_prob >= healthy_threshold:
        final_label = "Healthy"
        reason = (f"Final label: Healthy\n"
                  f"Healthy override is ON and met: Healthy {healthy_prob:.1%} ≥ Unhealthy {unhealthy_prob:.1%}, "
                  f"and ≥ threshold {healthy_threshold:.0%}. "
                  f"(Image top-1 would be {fused_label} {fused_prob:.1%}.)")
        healthy_override = True
    else:
        final_label = fused_label
        reason = (f"Final label: {fused_label}\n"
                  f"No Healthy override (Healthy {healthy_prob:.1%}, Unhealthy {unhealthy_prob:.1%}, "
                  f"threshold {healthy_threshold:.0%}).")
        healthy_override = False

    # Risk flags for guidance
    risk_flags: Set[str] = set()
    if any(k in fused_label.lower().replace(" ","").replace("_","") for k in CANCERY):
        risk_flags.add("cancer_like")
    if float(mc_top[0][1]) < LOW_CONF:
        risk_flags.add("system_uncertain")

    return dict(
        final_label=final_label,
        reason=reason,
        healthy_override=healthy_override,
        bin_top=bin_pairs,
        mc_top=mc_top,            # list of (label, prob) for Top-K
        fused_top=fused_top,
        bin_dict=bin_out,
        mc_dict={mc_labels[i]: float(mc_probs[i]) for i in range(len(mc_labels))},
        swap_binary_used=bool(swap_binary_order),
        healthy_threshold=float(healthy_threshold),
        prioritize_healthy=bool(prioritize_healthy),
        healthy_prob=float(healthy_prob),
        unhealthy_prob=float(unhealthy_prob),
        risk_flags=list(risk_flags),
    )

def rows_for_table(pairs: List[Tuple[str, float]]) -> List[Dict[str, str]]:
    return [{"label": l, "prob": f"{p:.3f}"} for (l, p) in pairs]

# === UPDATED: save into OUT_DIR and include final probability ===
def export_csv_json(final_label: str, info: Dict, image_name: str) -> Tuple[str, str]:
    # timestamp + output paths inside artifacts/reports/runs/
    t = _time_tag()
    csv_path = OUT_DIR / f"skinai_result_{t}.csv"
    json_path = OUT_DIR / f"skinai_result_{t}.json"

    # --- compute final probability ---
    # If Healthy override fired, use binary Healthy prob; else use fused top-1 (fallback to mc top-1)
    try:
        if info.get("healthy_override") and final_label == "Healthy":
            final_conf = float(info.get("healthy_prob", 0.0))
        else:
            if info.get("fused_top"):
                final_conf = float(info["fused_top"][0][1])
            else:
                final_conf = float(info["mc_top"][0][1])
    except Exception:
        final_conf = 0.0  # last-resort safety

    # --- build CSV rows (now with final prob) ---
    rows = [{"section": "final", "label": final_label, "prob": f"{final_conf:.6f}"}]

    for sec in ("bin_top", "mc_top", "fused_top"):
        for l, p in info[sec]:
            rows.append({"section": sec, "label": l, "prob": f"{float(p):.6f}"})

    # full binary probs
    for lbl, p in info["bin_dict"].items():
        rows.append({"section": "binary_probs", "label": lbl, "prob": f"{float(p):.6f}"})

    pd.DataFrame(rows).to_csv(csv_path, index=False)

    # --- JSON payload ---
    payload = dict(
        timestamp=t,
        image=image_name,
        final_label=final_label,
        final_confidence=final_conf,  # handy for downstream
        binary_probs=info["bin_dict"],
        image_top3=[{"label": l, "prob": float(p)} for l, p in info["mc_top"]],
        fused_top3=[{"label": l, "prob": float(p)} for l, p in info["fused_top"]],
        params=dict(
            HEALTHY_MIN_CONFIDENCE=info.get("healthy_threshold", HEALTHY_MIN_CONFIDENCE_DEFAULT),
            PRIORITIZE_BINARY_HEALTHY=info.get("prioritize_healthy", PRIORITIZE_BINARY_HEALTHY_DEFAULT),
            SWAP_BINARY_ORDER=info.get("swap_binary_used", False),
            TEMPERATURE=float(TEMPERATURE),
        ),
        reason=info.get("reason", ""),
        risk_flags=info.get("risk_flags", []),
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Gradio likes strings for downloadable paths
    return str(csv_path), str(json_path)


# =========================
# UI
# =========================
CSS = """
.gradio-container {max-width: 1160px !important;}
/* Headline spacing */
#hero h1 { margin-bottom: 6px !important; }
#hero p  { margin-top: 0 !important; color: #64748b; }
/* Result badge */
.result-badge { background: #16a34a; color: white; border-radius: 14px; padding: 12px 16px; font-weight: 800; font-size: 22px; display:inline-block; }
.result-badge.unhealthy { background:#1f2937; }
/* Subtle cards */
.card { border:1px solid #e5e7eb; background:#fff; border-radius:14px; box-shadow:0 1px 2px rgba(0,0,0,.04); padding:12px 16px; }
.dark .card { background:#0b1220; border-color:#374151; }
/* Guidance header & radio row */
.guidance-head { display:flex; align-items:center; justify-content:space-between; gap:10px; }
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo"), css=CSS) as demo:
    gr.Markdown(f"<div id='hero'><h1>{TITLE}</h1><p>{SUBTITLE}</p></div>")

    # State to carry Top-K info for guidance switching
    topk_state = gr.State(value=None)

    with gr.Row():
        with gr.Column(scale=1):
            inp_img = gr.Image(type="pil", label="Upload image", height=320)
            symptoms = gr.Textbox(label="(Optional) symptom text", placeholder="e.g., mild itch, no fever …", lines=3)

            analyze_btn = gr.Button("Analyze", variant="primary")
            clear_btn = gr.Button("Clear")

            with gr.Accordion("Settings", open=False):
                prio_ck = gr.Checkbox(value=PRIORITIZE_BINARY_HEALTHY_DEFAULT, label="Prioritize 'Healthy' when binary is confident")
                thr_sl  = gr.Slider(value=HEALTHY_MIN_CONFIDENCE_DEFAULT, minimum=0.5, maximum=0.9, step=0.01, label="Healthy override threshold")
                swap_ck = gr.Checkbox(value=SWAP_BINARY_DEFAULT, label="Swap Healthy/Unhealthy order (fix label-index mismatch)")

        with gr.Column(scale=2):
            final_label_html = gr.HTML()
            rationale_md = gr.Markdown()

            # Info card area (used for Healthy reassurance OR low-confidence neutral message)
            healthy_card = gr.HTML(visible=False)

            # Reveal buttons
            show_more_btn = gr.Button("Show other possibilities (Top-3)", visible=False)  # Healthy case
            show_guidance_btn = gr.Button("Show guidance anyway", visible=False)          # Low-confidence case

            # Guidance group (hidden by default when healthy; visible otherwise)
            with gr.Group(visible=False) as guidance_group:
                gr.HTML("<div class='guidance-head'><div class='card' style='display:inline-block; font-weight:600;'>View guidance for prediction</div></div>")
                guidance_pick = gr.Radio(choices=[], value=None, label="Choose Top-k", interactive=True)
                with gr.Row():
                    guidance_w = gr.HTML(label="Western (English)")
                    guidance_a = gr.HTML(label="Sinhala Ayurvedic (English)")

            with gr.Tabs():
                with gr.Tab("Confidence bars"):
                    bars_bin = gr.HTML()
                    bars_img = gr.HTML()
                    bars_fus = gr.HTML()
                with gr.Tab("Tables"):
                    tbl_bin = gr.Dataframe(headers=["label","prob"], interactive=False)
                    tbl_img = gr.Dataframe(headers=["label","prob"], interactive=False)
                    tbl_fus = gr.Dataframe(headers=["label","prob"], interactive=False)
                with gr.Tab("Raw JSON"):
                    js_bin = gr.JSON(label="Binary (Healthy/Unhealthy)")
                    js_img = gr.JSON(label="Image Top-3")
                    js_fus = gr.JSON(label="Fused Top-3")

            with gr.Row():
                exp_csv = gr.File(label="CSV export")
                exp_json = gr.File(label="JSON export")

    def _choices_from_topk(mc_top: List[Tuple[str,float]]) -> List[str]:
        out = []
        for idx, (lbl, p) in enumerate(mc_top[:TOPK], start=1):
            out.append(f"Top-{idx} — {lbl} ({p:.2f})")
        return out

    def _render_both_panels(selected_idx: int, state_payload: Dict) -> Tuple[str, str]:
        mc_top = state_payload["mc_top"]
        label, conf = mc_top[selected_idx]
        hp = state_payload["healthy_prob"]
        up = state_payload["unhealthy_prob"]
        thr = state_payload["healthy_threshold"]

        risk_flags = set(state_payload.get("risk_flags", []))
        if conf < LOW_CONF:
            risk_flags.add("system_uncertain")
        if any(k in label.lower().replace(" ","").replace("_","") for k in CANCERY):
            risk_flags.add("cancer_like")

        sug_w = build_suggestions(label, conf, risk_flags, hp, up, thr, STYLE_WESTERN)
        sug_a = build_suggestions(label, conf, risk_flags, hp, up, thr, STYLE_AYUR)
        return render_suggestions_html(sug_w), render_suggestions_html(sug_a)

    def do_analyze(pil_img, sym_text, prio_healthy, thr, swap_binary):
        if pil_img is None:
            # Reset everything cleanly
            return (
                "<div style='color:#ef4444;font-weight:700'>Please upload an image.</div>",
                "No image provided.",
                gr.update(visible=False, value=""),  # healthy_card (info card)
                gr.update(visible=False),            # show_more_btn
                gr.update(visible=False),            # show_guidance_btn
                gr.update(visible=False),            # guidance_group
                gr.update(choices=[], value=None), "", "",   # guidance_pick + both panels
                "", "", "",
                [], [], [],
                {}, [], [],
                None, None,
                None  # state
            )

        # --- Save uploaded image for debugging (safe, non-breaking) ---
        try:
            os.makedirs("artifacts/debug", exist_ok=True)
            stable_path = "artifacts/debug/last_input.jpg"
            pil_img.convert("RGB").save(stable_path, "JPEG", quality=92)
            print(f"[DEBUG] Saved last input: {stable_path}")
        except Exception as e:
            print(f"[DEBUG] Could not save debug image: {e}")

        info = predict_image(
            pil_img,
            swap_binary_order=bool(swap_binary),
            healthy_threshold=float(thr),
            prioritize_healthy=bool(prio_healthy),
        )
        final = info["final_label"]
        why = info["reason"]
        color = "#16a34a" if final.lower()=="healthy" else "#1f2937"
        pretty_final = final.replace("_", " ")
        final_html = f"<div class='card'><div class='result-badge{' unhealthy' if final.lower()!='healthy' else ''}'>{pretty_final}</div></div>"

        if info.get("swap_binary_used", False):
            why += "\n\n*(Note: Binary label order is swapped in Settings.)*"

        # Bars + tables + JSON
        bars_bin_html = _bar_html(info["bin_top"],  "Binary (Healthy vs Unhealthy)")
        bars_img_html = _bar_html(info["mc_top"],   "Image Top-3")
        bars_fus_html = _bar_html(info["fused_top"],"Fused Top-3")

        tb_bin = rows_for_table(info["bin_top"])
        tb_img = rows_for_table(info["mc_top"])
        tb_fus = rows_for_table(info["fused_top"])

        js_bin_obj = info["bin_dict"]
        js_img_obj = [{"label":l,"prob":p} for l,p in info["mc_top"]]
        js_fus_obj = [{"label":l,"prob":p} for l,p in info["fused_top"]]

        image_name = getattr(pil_img, "filename", "") or f"uploaded_{_time_tag()}.png"
        csv_path, json_path = export_csv_json(final, info, image_name)

        # Prepare shared state for guidance
        state_payload = dict(
            mc_top=[(str(l), float(p)) for l, p in info["mc_top"]],
            healthy_prob=float(info["healthy_prob"]),
            unhealthy_prob=float(info["unhealthy_prob"]),
            healthy_threshold=float(info["healthy_threshold"]),
            risk_flags=info.get("risk_flags", []),
        )

        # If Healthy override: show a small reassurance card + reveal button only.
        if info["healthy_override"] and final.lower() == "healthy":
            reassure = (
                f"<div class='card' style='background:#f0fdf4;border-color:#22c55e;'>"
                f"<div style='font-weight:700;margin-bottom:6px;'>✅ You're likely okay</div>"
                f"<div>Healthy override is ON and met. If anything changes (pain, fever, bleeding), you can always "
                f"re-run or consult a clinician.</div>"
                f"</div>"
            )
            return (
                final_html, why,
                gr.update(visible=True, value=reassure),   # healthy_card (info)
                gr.update(visible=True),                   # show_more_btn
                gr.update(visible=False),                  # show_guidance_btn
                gr.update(visible=False),                  # guidance_group hidden
                gr.update(choices=[], value=None), "", "", # guidance components cleared
                bars_bin_html, bars_img_html, bars_fus_html,
                tb_bin, tb_img, tb_fus,
                js_bin_obj, js_img_obj, js_fus_obj,
                csv_path, json_path,
                state_payload
            )

        # Otherwise (not-healthy): check low-confidence gate
        top1_label, top1_prob = info["mc_top"][0]
        if float(top1_prob) < HIGH_CONF_GATE:
            neutral = (
                "<div class='card' style='background:#eef2ff;border-color:#6366f1;'>"
                "<div style='font-weight:700;margin-bottom:6px;'>We’re not fully certain</div>"
                "<div>Lighting, focus, or framing can reduce accuracy. "
                "Try a close, well-lit photo in daylight and re-run. "
                "If you’re concerned, consult a clinician.</div>"
                "</div>"
            )
            # Keep guidance hidden; offer a button to reveal anyway
            return (
                final_html, why,
                gr.update(visible=True, value=neutral),    # healthy_card used as info card
                gr.update(visible=False),                  # show_more_btn
                gr.update(visible=True),                   # show_guidance_btn
                gr.update(visible=False),                  # guidance_group
                gr.update(choices=[], value=None), "", "", # guidance panels blank
                bars_bin_html, bars_img_html, bars_fus_html,
                tb_bin, tb_img, tb_fus,
                js_bin_obj, js_img_obj, js_fus_obj,
                csv_path, json_path,
                state_payload
            )

        # Not-healthy and confident enough: show guidance Top-1 immediately
        choices = _choices_from_topk(info["mc_top"])
        radio_update = gr.update(choices=choices, value=choices[0])
        gw_html, ga_html = _render_both_panels(0, state_payload)

        return (
            final_html, why,
            gr.update(visible=False, value=""),  # healthy_card hidden
            gr.update(visible=False),            # show_more_btn hidden
            gr.update(visible=False),            # show_guidance_btn hidden
            gr.update(visible=True),             # guidance_group visible
            radio_update, gw_html, ga_html,
            bars_bin_html, bars_img_html, bars_fus_html,
            tb_bin, tb_img, tb_fus,
            js_bin_obj, js_img_obj, js_fus_obj,
            csv_path, json_path,
            state_payload
        )

    analyze_btn.click(
        do_analyze,
        inputs=[inp_img, symptoms, prio_ck, thr_sl, swap_ck],
        outputs=[final_label_html, rationale_md,
                 healthy_card, show_more_btn, show_guidance_btn, guidance_group,
                 # inside guidance_group:
                 guidance_pick, guidance_w, guidance_a,
                 # visualizations:
                 bars_bin, bars_img, bars_fus,
                 tbl_bin, tbl_img, tbl_fus,
                 js_bin, js_img, js_fus,
                 exp_csv, exp_json,
                 topk_state],
        show_progress="full",
        api_name="analyze",
    )

    def on_pick_change(choice_text, state_payload):
        if not choice_text or not state_payload:
            return "", ""
        idx = 0
        if choice_text.startswith("Top-2"): idx = 1
        elif choice_text.startswith("Top-3"): idx = 2
        idx = max(0, min(idx, len(state_payload.get("mc_top", [])) - 1))
        return _render_both_panels(idx, state_payload)

    guidance_pick.change(
        on_pick_change,
        inputs=[guidance_pick, topk_state],
        outputs=[guidance_w, guidance_a]
    )

    def on_show_more(state_payload):
        if not state_payload:
            return gr.update(visible=False), gr.update(choices=[], value=None), "", ""
        choices = _choices_from_topk(state_payload["mc_top"])
        gw_html, ga_html = _render_both_panels(0, state_payload)
        return (
            gr.update(visible=True),                     # guidance_group visible
            gr.update(choices=choices, value=choices[0]),
            gw_html, ga_html
        )

    show_more_btn.click(
        on_show_more,
        inputs=[topk_state],
        outputs=[guidance_group, guidance_pick, guidance_w, guidance_a]
    )

    # Low-confidence: reveal guidance anyway
    show_guidance_btn.click(
        on_show_more,
        inputs=[topk_state],
        outputs=[guidance_group, guidance_pick, guidance_w, guidance_a]
    )

    def do_clear():
        return (
            None, "",                           # final + rationale
            gr.update(visible=False, value=""), # healthy_card (info)
            gr.update(visible=False),           # show_more_btn
            gr.update(visible=False),           # show_guidance_btn
            gr.update(visible=False),           # guidance_group
            gr.update(choices=[], value=None),  # guidance_pick
            "", "",                            # guidance_w/a
            "", "", "",                        # bars
            [], [], [],                         # tables
            {}, [], [],                         # raw JSONs
            None, None,                         # exports
            None                                # state
        )

    clear_btn.click(
        do_clear, inputs=[],
        outputs=[final_label_html, rationale_md,
                 healthy_card, show_more_btn, show_guidance_btn, guidance_group,
                 guidance_pick, guidance_w, guidance_a,
                 bars_bin, bars_img, bars_fus,
                 tbl_bin, tbl_img, tbl_fus,
                 js_bin, js_img, js_fus,
                 exp_csv, exp_json,
                 topk_state]
    )

if __name__ == "__main__":
    port = _pick_free_port(7860, max_tries=40)
    try:
        demo.launch(server_name="127.0.0.1", server_port=port or None, share=False)
    except Exception as e:
        print(f"[WARN] Localhost binding failed ({e}). Retrying with share=True on 0.0.0.0 …")
        demo.launch(server_name="0.0.0.0", server_port=port or None, share=True)
