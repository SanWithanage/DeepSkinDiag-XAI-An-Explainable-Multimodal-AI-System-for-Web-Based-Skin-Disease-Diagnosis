# backend/server.py
import os, io, json, datetime as dt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import torch
import torch.nn.functional as F
from PIL import Image
import timm
from torchvision import transforms, models as tvm
import pandas as pd

# =========================
# Config (keep in sync with app.py)
# =========================
TITLE = "SkinAI — API"
TOPK = 3
IMAGE_SIZE = 224
HEALTHY_MIN_CONFIDENCE_DEFAULT = 0.55
PRIORITIZE_BINARY_HEALTHY_DEFAULT = True
SWAP_BINARY_DEFAULT = False  # default UI toggle; request may override
HIGH_CONF_GATE = 0.60

PATH_BIN_CKPT   = "artifacts/checkpoints/binary/best.pt"
PATH_MC_CKPT    = "artifacts/checkpoints/multiclass/best.pt"
PATH_CALIB      = "artifacts/calibration/multiclass_temp.json"
PATH_LABELS_BIN = "artifacts/labels_binary.txt"
PATH_LABELS_26_CANDIDATES = ["labels_26.txt", "artifacts/labels_26.txt"]
PATH_GUIDANCE_JSON = "artifacts/guidance/guidance_en.json"

OUT_DIR = Path("artifacts/reports/runs"); OUT_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR = Path("artifacts/debug"); DEBUG_DIR.mkdir(parents=True, exist_ok=True)

STYLE_WESTERN = "Western (Dermatology) — English"
STYLE_AYUR    = "Sinhala Ayurvedic (Traditional) — English"
LOW_CONF = 0.55
CANCERY: Set[str] = {
    "skincancer","melanoma","basalcellcarcinoma","squamouscellcarcinoma","bcc","scc","cancer"
}

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
        if os.path.isfile(p): return p
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

def _variants_of_state_dict(sd: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
    keys = list(sd.keys()); variants = [sd]
    for prefix in ("module.", "model.", "net."):
        if all(k.startswith(prefix) for k in keys):
            variants.append({k[len(prefix):]: v for k, v in sd.items()})
    uniq, seen = [], set()
    for d in variants:
        marker = (len(d),) + tuple(sorted(list(d.keys())[:8]))
        if marker not in seen: uniq.append(d); seen.add(marker)
    return uniq

def _shape_filtered(sd: Dict[str, torch.Tensor], model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    m_sd = model.state_dict(); out = {}
    for k, v in sd.items():
        if k in m_sd and hasattr(v, "shape") and hasattr(m_sd[k], "shape") and tuple(v.shape) == tuple(m_sd[k].shape):
            out[k] = v
    return out

def _extract_labelnames_from_ckpt(d: dict) -> Optional[List[str]]:
    for k in ("labels","class_names","classes","idx_to_class"):
        if k in d:
            v = d[k]
            if isinstance(v, dict):
                try: v = [v[i] for i in sorted(v, key=lambda x: int(x))]
                except Exception: v = list(v.values())
            if isinstance(v, (list, tuple)) and all(isinstance(x, (str, int)) for x in v):
                return [str(x) for x in v]
    return None

def _bar_items(pairs: List[Tuple[str, float]]) -> List[Dict[str, float]]:
    return [{"label": l, "prob": float(p)} for (l, p) in pairs]

# =========================
# Labels + calibration
# =========================
if os.path.isfile(PATH_LABELS_BIN):
    bin_labels_file = _read_lines(PATH_LABELS_BIN)
else:
    bin_labels_file = ["Healthy","Unhealthy"]
assert "Healthy" in bin_labels_file and "Unhealthy" in bin_labels_file, \
    f"{PATH_LABELS_BIN} must contain 'Healthy' and 'Unhealthy' (one per line)."

TEMPERATURE = _load_temperature(PATH_CALIB)
labels_26_path = _first_existing(PATH_LABELS_26_CANDIDATES)
mc_labels_file = _read_lines(labels_26_path) if labels_26_path else None

# =========================
# Preprocessing & device
# =========================
img_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])
device = _device()
print(f"[INFO] Using device: {device.type}")

# =========================
# Load binary model
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
            bin_model.load_state_dict(filtered, strict=False)
            print(f"[OK] Binary weights loaded ({len(filtered)} matched keys)."); loaded_bin = True; break
        except Exception: pass
if not loaded_bin:
    print("[WARN] Binary checkpoint not matched by shape. Using random head.")
bin_model.eval().to(device)

# =========================
# Multiclass model (auto detect)
# =========================
mc_ckpt = torch.load(PATH_MC_CKPT, map_location="cpu")
ckpt_labels = None; arch_hint = None; raw_state = None
if isinstance(mc_ckpt, dict):
    ckpt_labels = _extract_labelnames_from_ckpt(mc_ckpt)
    if "arch" in mc_ckpt and isinstance(mc_ckpt["arch"], str): arch_hint = mc_ckpt["arch"]
    if "state_dict" in mc_ckpt and isinstance(mc_ckpt["state_dict"], dict): raw_state = mc_ckpt["state_dict"]
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
        raise RuntimeError("Could not find multiclass labels (ckpt or labels_26.txt).")

def _build_candidates(num_classes: int, hint: Optional[str]):
    cands = []
    if hint:
        try: cands.append((timm.create_model(hint, pretrained=False, num_classes=num_classes), f"timm:{hint}"))
        except Exception: pass
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
    for a in ["tf_efficientnet_b0","efficientnet_b0","mobilenetv3_large_100","resnet34","convnext_tiny"]:
        try: cands.append((timm.create_model(a, pretrained=False, num_classes=num_classes), f"timm:{a}"))
        except Exception: pass
    try:
        m = tvm.efficientnet_b0(weights=None); in_feat = m.classifier[1].in_features
        m.classifier[1] = torch.nn.Linear(in_feat, num_classes)
        cands.append((m, "torchvision:efficientnet_b0"))
    except Exception: pass
    try:
        m = tvm.resnet50(weights=None); in_feat = m.fc.in_features
        m.fc = torch.nn.Linear(in_feat, num_classes)
        cands.append((m, "torchvision:resnet50"))
    except Exception: pass
    return cands

def _best_model_for_ckpt(num_classes: int, arch_hint: Optional[str], raw_state: Dict[str, torch.Tensor]):
    sd_variants = _variants_of_state_dict(raw_state) if isinstance(raw_state, dict) else []
    if not sd_variants: raise RuntimeError("Checkpoint has no state_dict dict.")
    candidates = _build_candidates(num_classes, arch_hint)
    if not candidates: raise RuntimeError("No candidate backbones.")
    best = None
    for model, tag in candidates:
        m_sd = model.state_dict()
        for v_idx, sdv in enumerate(sd_variants):
            filt = {k: v for k, v in sdv.items() if k in m_sd and tuple(v.shape) == tuple(m_sd[k].shape)}
            match = len(filt)
            if best is None or match > best["match"]:
                best = dict(model=model, tag=tag, sd_variant=v_idx, matched_keys=filt, match=match, total=len(sdv))
    if best is None or best["match"] == 0:
        raise RuntimeError("Could not match any parameters by shape for the multiclass checkpoint.")
    best["model"].load_state_dict(best["matched_keys"], strict=False)
    print(f"[OK] Multiclass backbone: {best['tag']} (sd variant #{best['sd_variant']}) – matched {best['match']} keys.")
    return best["model"], best

mc_model, _best = _best_model_for_ckpt(num_classes=len(mc_labels), arch_hint=arch_hint, raw_state=raw_state)
mc_model.eval().to(device)

# =========================
# Guidance (same structure as app.py)
# =========================
GENERIC_WESTERN = [
    "Photograph the area in daylight every 2–3 days to track changes.",
    "Use a gentle, fragrance-free moisturizer; avoid harsh scrubs and peels.",
    "Use broad-spectrum sunscreen on exposed areas during the day.",
    "Patch-test any new product for 24 hours; stop if irritation appears.",
]
GENERIC_AYUR = [
    "Keep the area clean and dry; avoid friction and tight clothing.",
    "Use a cool compress (clean cloth, cool boiled water) for comfort.",
    "Patch-test any new natural preparation (small area, 24h).",
    "If symptoms worsen or persist, consult a professional.",
]

GUIDE = None
if os.path.isfile(PATH_GUIDANCE_JSON):
    try:
        with open(PATH_GUIDANCE_JSON, "r", encoding="utf-8") as f:
            GUIDE = json.load(f)
        assert isinstance(GUIDE, dict) and "western" in GUIDE and "ayur" in GUIDE, "Invalid guidance JSON."
        assert "generic" in GUIDE["western"] and "generic" in GUIDE["ayur"], "Guidance missing 'generic'."
        print(f"[GUIDE] Loaded {PATH_GUIDANCE_JSON}")
    except Exception as e:
        print(f"[GUIDE] Failed to load guidance: {e}")
        GUIDE = None
else:
    print("[GUIDE] Not found – using built-ins.")

def get_tips(style: str, top_label: str) -> List[str]:
    use_western = style.startswith("Western")
    if GUIDE:
        bucket = GUIDE["western"] if use_western else GUIDE["ayur"]
        return bucket.get(top_label, bucket["generic"])
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
            "Some features look concerning. Seek a clinician’s advice soon—"
            "especially if the spot changes, bleeds, or you feel unwell."
        )
    elif triage == "amber":
        banner, detail = "⚠️ Caution advised", (
            "The model is not fully certain. Monitor closely, reduce irritants, and consider a visit "
            "if it persists beyond 1–2 weeks or worsens."
        )
    else:
        banner, detail = "✅ Looks mild", "Gentle skin-care steps may help. Keep monitoring for any changes."
    if triage == "red":
        next_steps = [
            "Book a dermatology appointment as soon as possible.",
            "Bring clear photos taken in daylight over several days.",
            "Note symptom timeline, triggers, and any pain/fever/bleeding.",
        ]
    elif triage == "amber":
        next_steps = [
            "If no improvement in 1–2 weeks or rapid spread, see a clinician.",
            "Minimize friction/irritants; keep the area clean and moisturized.",
        ]
    else:
        next_steps = [
            "Follow gentle care for 7–10 days and reassess.",
            "If severe pain, fever, or bleeding appears, escalate to a clinician.",
        ]
    tips = get_tips(style, top_label)
    conf_line = (f"Model confidence: {top_conf:.2f} (top: {top_label}) · "
                 f"Healthy {healthy_prob:.2f} vs Unhealthy {unhealthy_prob:.2f} (thr {healthy_threshold:.2f})")
    disclaimer = ("This app provides guidance only and is not a medical diagnosis. "
                  "Always consult a qualified professional for medical concerns.")
    return dict(
        triage=triage, banner=banner, banner_detail=detail,
        confidence=conf_line, next_steps=next_steps, tips=tips, disclaimer=disclaimer
    )

# =========================
# Inference
# =========================
@torch.inference_mode()
def predict_image(img: Image.Image, swap_binary_order: bool, healthy_threshold: float, prioritize_healthy: bool) -> Dict:
    x = img_tf(img.convert("RGB")).unsqueeze(0).to(device)

    # Binary
    bin_logits = bin_model(x)
    bin_probs_raw = _softmax(bin_logits, T=1.0).squeeze(0).cpu()
    file_order = list(bin_labels_file)            # e.g., ["Healthy","Unhealthy"]
    file_swapped = [file_order[1], file_order[0]]
    chosen_order = file_swapped if swap_binary_order else file_order
    bin_out = {chosen_order[i]: float(bin_probs_raw[i]) for i in range(2)}
    healthy_prob = bin_out.get("Healthy", 0.0)
    unhealthy_prob = bin_out.get("Unhealthy", 0.0)
    bin_pairs = sorted(bin_out.items(), key=lambda z: z[1], reverse=True)[:TOPK]

    # Multiclass
    mc_logits = mc_model(x)
    mc_probs = _softmax(mc_logits, T=TEMPERATURE).squeeze(0).cpu()
    mc_dict = {mc_labels[i]: float(mc_probs[i]) for i in range(len(mc_labels))}
    mc_top = sorted(mc_dict.items(), key=lambda z: z[1], reverse=True)[:TOPK]
    fused_top = mc_top[:]

    # Healthy override
    fused_label, fused_prob = fused_top[0]
    if prioritize_healthy and healthy_prob >= unhealthy_prob and healthy_prob >= healthy_threshold:
        final_label = "Healthy"
        reason = (f"Final: Healthy — override ON and met: "
                  f"Healthy {healthy_prob:.1%} ≥ Unhealthy {unhealthy_prob:.1%}, "
                  f"≥ thr {healthy_threshold:.0%}. (Image top-1 would be {fused_label} {fused_prob:.1%}.)")
        healthy_override = True
    else:
        final_label = fused_label
        reason = (f"Final: {fused_label} — no override (Healthy {healthy_prob:.1%}, Unhealthy {unhealthy_prob:.1%}, "
                  f"thr {healthy_threshold:.0%}).")
        healthy_override = False

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
        mc_top=mc_top,
        fused_top=fused_top,
        bin_dict=bin_out,
        mc_dict=mc_dict,
        healthy_prob=float(healthy_prob),
        unhealthy_prob=float(unhealthy_prob),
        healthy_threshold=float(healthy_threshold),
        risk_flags=list(risk_flags),
    )

def export_csv_json(final_label: str, info: Dict, image_name: str) -> Tuple[str, str, str]:
    run_id = _time_tag()
    csv_path = OUT_DIR / f"skinai_result_{run_id}.csv"
    json_path = OUT_DIR / f"skinai_result_{run_id}.json"
    # final confidence
    if info.get("healthy_override") and final_label == "Healthy":
        final_conf = float(info.get("healthy_prob", 0.0))
    else:
        final_conf = float((info.get("fused_top") or info.get("mc_top"))[0][1])
    rows = [{"section": "final", "label": final_label, "prob": f"{final_conf:.6f}"}]
    for sec in ("bin_top","mc_top","fused_top"):
        for l, p in info[sec]:
            rows.append({"section": sec, "label": l, "prob": f"{float(p):.6f}"})
    for lbl, p in info["bin_dict"].items():
        rows.append({"section": "binary_probs", "label": lbl, "prob": f"{float(p):.6f}"})
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    payload = dict(
        run_id=run_id,
        timestamp=run_id,
        image=image_name,
        final_label=final_label,
        final_confidence=final_conf,
        binary_probs=info["bin_dict"],
        image_top3=[{"label": l, "prob": float(p)} for l, p in info["mc_top"]],
        fused_top3=[{"label": l, "prob": float(p)} for l, p in info["fused_top"]],
        params=dict(
            HEALTHY_MIN_CONFIDENCE=info.get("healthy_threshold", HEALTHY_MIN_CONFIDENCE_DEFAULT),
            PRIORITIZE_BINARY_HEALTHY=bool(final_label=="Healthy" and info.get("healthy_override", False) or True),
            SWAP_BINARY_ORDER=False,  # will be echoed from request below
            TEMPERATURE=float(TEMPERATURE),
        ),
        reason=info.get("reason",""),
        risk_flags=info.get("risk_flags", []),
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return run_id, str(csv_path), str(json_path)

# =========================
# FastAPI app
# =========================
app = FastAPI(title=TITLE)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        # local dev
        "http://localhost:5173", "https://localhost:5173",
        "http://127.0.0.1:5173", "https://127.0.0.1:5173",

        # your LAN dev origin(s) so your phone can reach the frontend
        "http://10.230.172.64:5173", "https://10.230.172.64:5173",
        # sometimes Vite uses 5174
        "http://10.230.172.64:5174", "https://10.230.172.64:5174",

        # (add prod frontend(s) after deploy)
        # "https://skinai.yourdomain.com",
        # "https://skinai.netlify.app",
    ],  # during dev; lock down in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
def health():
    return {"ok": True, "ts": _time_tag()}

class AnalyzeResponse(BaseModel):
    run_id: str
    final_label: str
    reason: str
    healthy_override: bool
    healthy_prob: float
    unhealthy_prob: float
    topk_binary: List[Dict]
    topk_image: List[Dict]
    topk_fused: List[Dict]
    risk_flags: List[str]
    params_echo: Dict
    download_csv_url: str
    download_json_url: str
    low_conf_gate_triggered: bool

@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(
    image: UploadFile = File(...),
    prioritize_healthy: bool = Form(PRIORITIZE_BINARY_HEALTHY_DEFAULT),
    healthy_threshold: float = Form(HEALTHY_MIN_CONFIDENCE_DEFAULT),
    swap_binary_order: bool = Form(SWAP_BINARY_DEFAULT),
    symptom_text: Optional[str] = Form(None),
):
    # Load image
    try:
        raw = await image.read()
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
        # save debug
        try:
            dbg_path = DEBUG_DIR / "last_input.jpg"
            pil.save(dbg_path, "JPEG", quality=92)
        except Exception as e:
            print(f"[DEBUG] Could not save debug image: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # Inference
    info = predict_image(
        pil,
        swap_binary_order=bool(swap_binary_order),
        healthy_threshold=float(healthy_threshold),
        prioritize_healthy=bool(prioritize_healthy),
    )

    # Export CSV/JSON
    image_name = image.filename or f"uploaded_{_time_tag()}.png"
    run_id, csv_path, json_path = export_csv_json(info["final_label"], info, image_name)

    # Embed request params in payload
    info_params = dict(
        HEALTHY_MIN_CONFIDENCE=float(healthy_threshold),
        PRIORITIZE_BINARY_HEALTHY=bool(prioritize_healthy),
        SWAP_BINARY_ORDER=bool(swap_binary_order),
        TEMPERATURE=float(TEMPERATURE),
    )

    # Low-confidence UI gate (for front-end to decide hiding guidance)
    top1_label, top1_prob = info["mc_top"][0]
    low_conf_gate_triggered = float(top1_prob) < HIGH_CONF_GATE and (info["final_label"].lower() != "healthy")

    return AnalyzeResponse(
        run_id=run_id,
        final_label=info["final_label"],
        reason=info["reason"],
        healthy_override=bool(info["healthy_override"]),
        healthy_prob=float(info["healthy_prob"]),
        unhealthy_prob=float(info["unhealthy_prob"]),
        topk_binary=_bar_items(info["bin_top"]),
        topk_image=_bar_items(info["mc_top"]),
        topk_fused=_bar_items(info["fused_top"]),
        risk_flags=list(info.get("risk_flags", [])),
        params_echo=info_params,
        download_csv_url=f"/api/runs/{run_id}/csv",
        download_json_url=f"/api/runs/{run_id}/json",
        low_conf_gate_triggered=low_conf_gate_triggered,
    )

def _safe_get_run_file(run_id: str, ext: str) -> Path:
    p = OUT_DIR / f"skinai_result_{run_id}.{ext}"
    if not p.exists(): raise HTTPException(status_code=404, detail="Run not found.")
    return p

@app.get("/api/runs/{run_id}/csv")
def get_csv(run_id: str):
    p = _safe_get_run_file(run_id, "csv")
    return FileResponse(str(p), media_type="text/csv", filename=p.name)

@app.get("/api/runs/{run_id}/json")
def get_json(run_id: str):
    p = _safe_get_run_file(run_id, "json")
    return FileResponse(str(p), media_type="application/json", filename=p.name)

@app.get("/")
def root():
    return PlainTextResponse("SkinAI API is running. POST /api/analyze")
