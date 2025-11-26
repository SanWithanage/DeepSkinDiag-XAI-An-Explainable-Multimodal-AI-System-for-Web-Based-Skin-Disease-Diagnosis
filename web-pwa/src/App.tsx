import React, { useEffect, useMemo, useRef, useState } from "react";
import { API } from "./api";
import type { AnalyzeResponse, BarItem } from "./types";
import { enqueue, listQueue, remove } from "./offlineQueue";
import { motion, AnimatePresence } from "framer-motion";
import {
  Camera,
  Upload,
  Loader2,
  CheckCircle2,
  AlertTriangle,
  AlertOctagon,
  Download,
  SlidersHorizontal,
  Eye,
  EyeOff,
} from "lucide-react";

/* ---------- Small atoms ---------- */
const Card: React.FC<React.PropsWithChildren<{ className?: string }>> = ({ children, className = "" }) => (
  <div className={`card ${className}`}>{children}</div>
);

type PillTone = "good" | "warn" | "bad";
const Pill: React.FC<{ text: string; tone: PillTone }> = ({ text, tone }) => {
  const color = tone === "good" ? "bg-emerald-600" : tone === "warn" ? "bg-amber-600" : "bg-rose-600";
  const Icon = tone === "good" ? CheckCircle2 : tone === "warn" ? AlertTriangle : AlertOctagon;
  return (
    <span className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-white text-sm font-bold ${color}`}>
      <Icon size={16} /> {text}
    </span>
  );
};

const Bar: React.FC<{ item: BarItem }> = ({ item }) => {
  const pct = Math.round(item.prob * 100);
  return (
    <div className="my-2">
      <div className="flex items-center justify-between text-sm">
        <span className="truncate mr-3" title={item.label}>{item.label.replaceAll("_", " ")}</span>
        <span className="tabular-nums">{pct}%</span>
      </div>
      <div className="relative h-2.5 rounded-full bg-gray-200 dark:bg-gray-700 overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ type: "spring", stiffness: 120, damping: 20 }}
          className="absolute inset-y-0 left-0 rounded-full"
          style={{ background: "linear-gradient(90deg,#6366f1,#22c55e)" }}
        />
      </div>
    </div>
  );
};

/* Animated radial gauge (no extra deps) */
const Gauge: React.FC<{ value: number; label?: string }> = ({ value, label }) => {
  const pct = Math.max(0, Math.min(1, value));
  return (
    <div className="relative w-28 h-28">
      <motion.div
        className="absolute inset-0 rounded-full"
        style={{ background: `conic-gradient(#10b981 ${pct * 360}deg, rgba(0,0,0,.08) 0)` }}
        initial={{ rotate: -90 }}
        animate={{ rotate: 0 }}
        transition={{ type: "spring", stiffness: 120, damping: 18 }}
      />
      <div className="absolute inset-2 rounded-full bg-white/80 dark:bg-gray-900/80 shadow-inner grid place-items-center">
        <div className="text-lg font-extrabold">{Math.round(pct * 100)}%</div>
        {label && <div className="text-xs opacity-60 -mt-1">{label}</div>}
      </div>
    </div>
  );
};

/* ---------- Main App ---------- */
export default function App() {
  // form
  const [file, setFile] = useState<File | null>(null);
  const [symptomText, setSymptomText] = useState("");
  const [prioritizeHealthy, setPrioritizeHealthy] = useState(true);
  const [healthyThr, setHealthyThr] = useState(0.55);
  const [swapBinary, setSwapBinary] = useState(false);

  // app
  const [busy, setBusy] = useState(false);
  const [res, setRes] = useState<AnalyzeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const [backendOK, setBackendOK] = useState<boolean | null>(null);
  const [queued, setQueued] = useState<number>(0);

  const inputRef = useRef<HTMLInputElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const [choiceIdx, setChoiceIdx] = useState(0);
  const [showPoss, setShowPoss] = useState(false);

  useEffect(() => {
    API.health().then(() => setBackendOK(true)).catch(() => setBackendOK(false));
    listQueue().then((q) => setQueued(q.length)).catch(() => {});
  }, []);

  useEffect(() => {
    async function process() {
      if (!navigator.onLine) return;
      const jobs = await listQueue();
      for (const j of jobs) {
        try {
          const f = new File([j.file], j.filename, { type: (j.file as any).type || "image/jpeg" });
          await API.analyze({
            file: f,
            prioritize_healthy: j.prioritize_healthy,
            healthy_threshold: j.healthy_threshold,
            swap_binary_order: j.swap_binary_order,
            symptom_text: j.symptom_text,
          });
          await remove(j.id);
        } catch { break; }
      }
      const left = (await listQueue()).length; setQueued(left);
      if (left === 0 && jobs.length > 0) setToast("Queued analyses uploaded.");
    }
    window.addEventListener("online", process);
    process();
    return () => window.removeEventListener("online", process);
  }, []);

  function onPick(fs: FileList | null) {
    const f = fs?.[0] || null;
    setFile(f); setRes(null); setError(null); setShowPoss(false);
    if (f && imgRef.current) imgRef.current.src = URL.createObjectURL(f);
  }

  async function onAnalyze() {
    if (!file) { setToast("Please choose an image first."); return; }

    if (!navigator.onLine) {
      await enqueue({
        file, filename: file.name || "photo.jpg",
        prioritize_healthy: prioritizeHealthy, healthy_threshold: healthyThr,
        swap_binary_order: swapBinary, symptom_text: symptomText || undefined,
      }).catch(() => {});
      setQueued((q) => q + 1); setToast("No internet. Photo queued — will auto-upload when online.");
      return;
    }

    setBusy(true); setError(null); setRes(null); setShowPoss(false);
    try {
      const r = await API.analyze({
        file, prioritize_healthy: prioritizeHealthy,
        healthy_threshold: healthyThr, swap_binary_order: swapBinary,
        symptom_text: symptomText || undefined,
      });
      setRes(r); setChoiceIdx(0);
    } catch (e: any) { setError(String(e?.message ?? e)); }
    finally { setBusy(false); }
  }

  function resetAll() {
    setFile(null); setRes(null); setError(null); setChoiceIdx(0); setShowPoss(false);
    if (imgRef.current) imgRef.current.src = "";
    if (inputRef.current) inputRef.current.value = "";
  }

  // derived
  const choices = useMemo(
    () => (res?.topk_image ?? []).slice(0, 3).map((x, i) => ({ idx: i, txt: `Top-${i + 1} — ${x.label} (${x.prob.toFixed(2)})` })),
    [res]
  );
  const top = res?.topk_image?.[choiceIdx] ?? null;

  const { binHealthy, binUnhealthy } = useMemo(() => {
    const h = res?.topk_binary.find(b => b.label.toLowerCase() === "healthy")?.prob ?? 0;
    const u = res?.topk_binary.find(b => b.label.toLowerCase() === "unhealthy")?.prob ?? 0;
    return { binHealthy: h, binUnhealthy: u };
  }, [res]);

  const clientHealthy = useMemo(() => {
    if (!res || !prioritizeHealthy) return false;
    return binHealthy >= binUnhealthy && binHealthy >= healthyThr;
  }, [res, prioritizeHealthy, binHealthy, binUnhealthy, healthyThr]);

  const shouldShowGuidance =
    !!res && !clientHealthy &&
    (res.final_label.toLowerCase() !== "healthy" || res.low_conf_gate_triggered);

  /* ---------- UI ---------- */
  return (
    <div className="min-h-screen text-gray-900 dark:text-gray-100 font-sans">
      {/* Toast */}
      <AnimatePresence>
        {toast && (
          <motion.div initial={{ y: 60, opacity: 0 }} animate={{ y: 0, opacity: 1 }} exit={{ y: 60, opacity: 0 }}
            transition={{ type: "spring", bounce: 0 }}
            className="fixed bottom-[max(1rem,env(safe-area-inset-bottom))] left-0 right-0 z-50 flex justify-center">
            <div className="card bg-black/80 text-white"><div className="px-4 py-2">{toast}</div></div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Busy overlay */}
      <AnimatePresence>
        {busy && (
          <motion.div className="fixed inset-0 z-40 bg-black/30 backdrop-blur-sm grid place-items-center"
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <div className="card px-6 py-4 flex items-center gap-2"><Loader2 className="animate-spin" /> Analyzing…</div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Header */}
      <div className="max-w-5xl mx-auto px-4 pt-6 pb-2">
        <div className="flex items-center justify-between gap-3">
          <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
            <h1 className="text-3xl sm:text-4xl font-black tracking-tight">SkinAI</h1>
            <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">Image Diagnosis (with Healthy Override)</p>
          </motion.div>
          {queued > 0 && <div className="text-xs bg-amber-600 text-white px-2 py-1 rounded-lg">{queued} queued</div>}
        </div>
        {backendOK === false && (
          <div className="mt-2 text-sm text-rose-600">Backend unreachable. Start FastAPI on 127.0.0.1:8000 or check the proxy.</div>
        )}
      </div>

      {/* Main layout */}
      <div className="max-w-5xl mx-auto px-4 pb-[max(1rem,env(safe-area-inset-bottom))] grid md:grid-cols-3 gap-4">
        {/* Left */}
        <motion.div initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} className="space-y-3 md:col-span-1">
          <Card>
            <div className="font-semibold text-sm mb-2">Upload image</div>
            <div
              className="rounded-2xl border-2 border-dashed border-gray-300 dark:border-gray-700 p-4 text-center cursor-pointer hover:bg-black/5 dark:hover:bg-white/5 transition"
              onDragOver={(e)=>e.preventDefault()}
              onDrop={(e)=>{e.preventDefault(); onPick(e.dataTransfer?.files||null);}}
              onClick={()=>inputRef.current?.click()}
            >
              <div className="flex items-center justify-center gap-2 text-sm opacity-80"><Camera size={16}/> Drag & drop or tap to choose / take a photo</div>
              <div className="text-xs opacity-60">JPG/PNG • clear daylight photo • close focus</div>
            </div>
            <input ref={inputRef} type="file" accept="image/*" capture="environment" className="hidden" onChange={(e)=>onPick(e.target.files)} />
            <img ref={imgRef} alt="" className="mt-3 rounded-xl w-full object-cover max-h-72 animate-fadeUp" />
            <div className="mt-3 grid grid-cols-2 gap-2">
              <button onClick={onAnalyze} disabled={busy || !file} className={`btn-primary ${busy || !file ? "opacity-60" : ""}`}>
                <Upload size={16}/> {busy?"Analyzing…":"Analyze"}
              </button>
              <button onClick={resetAll} className="btn-outline">Clear</button>
            </div>
          </Card>

          <Card>
            <div className="font-semibold text-sm mb-2">(Optional) symptom text</div>
            <textarea value={symptomText} onChange={(e)=>setSymptomText(e.target.value)} placeholder="e.g., mild itch, no fever …" className="input min-h-[96px]" />
          </Card>

          <Card>
            <details>
              <summary className="cursor-pointer font-semibold flex items-center gap-2"><SlidersHorizontal size={16}/> Settings</summary>
              <div className="mt-3 space-y-3">
                <label className="flex items-center gap-2 select-none"><input type="checkbox" className="accent-brand-600 w-5 h-5" checked={prioritizeHealthy} onChange={(e)=>setPrioritizeHealthy(e.target.checked)} /> Prioritize “Healthy” when binary is confident</label>
                <div>
                  <div className="text-sm font-semibold">Healthy override threshold: {healthyThr.toFixed(2)}</div>
                  <input type="range" min={0.5} max={0.9} step={0.01} value={healthyThr} onChange={(e)=>setHealthyThr(parseFloat(e.target.value))} className="w-full" />
                </div>
                <label className="flex items-center gap-2 select-none"><input type="checkbox" className="accent-brand-600 w-5 h-5" checked={swapBinary} onChange={(e)=>setSwapBinary(e.target.checked)} /> Swap Healthy/Unhealthy order (fix label-index mismatch)</label>
              </div>
            </details>
          </Card>
        </motion.div>

        {/* Right */}
        <motion.div initial={{ opacity: 0, x: 10 }} animate={{ opacity: 1, x: 0 }} className="space-y-3 md:col-span-2">
          <Card>
            {!res ? (
              <div className="text-sm text-gray-600 dark:text-gray-400">No result yet.</div>
            ) : (
              <div className="space-y-3">
                <div className="flex items-center justify-between flex-wrap gap-2">
                  <Pill
                    text={(clientHealthy ? "Healthy" : res.final_label).replaceAll("_"," ")}
                    tone={(clientHealthy || res.final_label.toLowerCase()==="healthy") ? "good" : "bad"}
                  />
                  {clientHealthy && (
                    <button className="btn-outline !py-1 !h-9" onClick={()=>setShowPoss(s=>!s)}>
                      {showPoss ? <EyeOff size={16}/> : <Eye size={16}/> } {showPoss ? "Hide possibilities" : "Show other possibilities"}
                    </button>
                  )}
                  {error && <div className="text-rose-600 text-sm">{error}</div>}
                </div>

                {/* Healthy summary row */}
                {(clientHealthy || (res.final_label.toLowerCase()==="healthy" && res.healthy_override)) && !showPoss && (
                  <div className="flex items-center gap-4">
                    <Gauge value={binHealthy} label="Healthy" />
                    <div className="flex-1">
                      <div className="font-bold mb-0.5">You’re likely okay</div>
                      <div className="text-sm opacity-80">If anything changes (pain, fever, bleeding), re-run or consult a clinician.</div>
                    </div>
                  </div>
                )}

                <pre className="text-sm whitespace-pre-wrap leading-relaxed">
                  {clientHealthy
                    ? `Final: Healthy — client Healthy override met (Healthy ${(binHealthy*100).toFixed(1)}% ≥ Unhealthy ${(binUnhealthy*100).toFixed(1)}%, ≥ thr ${(healthyThr*100).toFixed(0)}%).`
                    : res.reason}
                </pre>
              </div>
            )}
          </Card>

          {/* Guidance (hidden if clientHealthy unless toggled) */}
          {(res && (showPoss || (!clientHealthy && (res.final_label.toLowerCase()!=="healthy" || res.low_conf_gate_triggered)))) && (
            <Card>
              <div className="flex items-center justify-between flex-wrap gap-2">
                <div className="font-semibold">View guidance for prediction</div>
                <select value={choiceIdx} onChange={(e)=>setChoiceIdx(parseInt(e.target.value))} className="input py-1 h-9 max-w-[280px]">
                  {choices.map(c => <option key={c.idx} value={c.idx}>{c.txt}</option>)}
                </select>
              </div>

              {/* Two compact guidance panels with calm colors */}
              <div className="mt-3 grid md:grid-cols-2 gap-3">
                {(["Western (English)", "Sinhala Ayurvedic — English"] as const).map((name, i) => {
                  const tips = i===0 ? [
                    "Photograph the area in daylight every 2–3 days.",
                    "Use a gentle, fragrance-free moisturizer; avoid harsh scrubs/peels.",
                    "Use broad-spectrum sunscreen on exposed areas.",
                    "Patch-test new products for 24h; stop if irritated.",
                  ] : [
                    "Keep the area clean and dry; avoid friction and tight clothing.",
                    "Use a cool compress (clean cloth, cooled boiled water).",
                    "Patch-test natural preparations (small area, 24h).",
                    "Consult a professional if symptoms persist or worsen.",
                  ];
                  const top = res!.topk_image[choiceIdx];
                  const conf = top?.prob ?? 0;
                  const tone = (conf < 0.55 || res!.low_conf_gate_triggered) ? ("warn" as const) : ("good" as const);
                  const ring = tone==="good"?"border-emerald-500/60 bg-emerald-50/60 dark:bg-emerald-950/30":"border-amber-500/60 bg-amber-50/60 dark:bg-amber-950/30";
                  const hp = binHealthy, up = binUnhealthy;
                  const confLine = `Model confidence: ${conf.toFixed(2)} (top: ${top?.label}) · Healthy ${hp.toFixed(2)} vs Unhealthy ${up.toFixed(2)} (thr ${healthyThr.toFixed(2)})`;
                  const next = tone==="warn"
                    ? ["If no improvement in 1–2 weeks or rapid spread, see a clinician.","Minimize friction/irritants; keep area clean and moisturized."]
                    : ["Follow gentle care for 7–10 days and reassess.","Escalate if severe pain, fever, or bleeding appears."];
                  const title = tone==="good"?"✅ Looks mild":"⚠️ Caution advised";
                  return (
                    <div key={i} className={`card border-2 ${ring} animate-fadeUp`}>
                      <div className="font-extrabold mb-1">{title}</div>
                      <div className="mb-3">{tone==="good" ? "General skin-care steps may help. Keep monitoring for any changes." : "Model not fully certain. Monitor closely and consider a dermatology visit if it persists/worsens."}</div>
                      <div className="mb-3"><b>Confidence:</b> {confLine}</div>
                      <div className="font-semibold mb-1">What to do next</div>
                      <ul className="list-disc ml-5 mb-3">{next.map((x,idx)=>(<li key={idx} className="my-1">{x}</li>))}</ul>
                      <div className="font-semibold mb-1">Care tips ({name})</div>
                      <ul className="list-disc ml-5 mb-2">{tips.map((x,idx)=>(<li key={idx} className="my-1">{x}</li>))}</ul>
                      <div className="text-sm opacity-80 italic">This app provides guidance only and is not a medical diagnosis.</div>
                    </div>
                  );
                })}
              </div>
            </Card>
          )}

          {/* Visualizations */}
          {res && (
            <Card>
              <div className="flex flex-wrap gap-2 mb-3">
                <a className="btn-outline" href={res.download_csv_url} target="_blank" rel="noreferrer"><Download size={16}/> CSV export</a>
                <a className="btn-outline" href={res.download_json_url} target="_blank" rel="noreferrer"><Download size={16}/> JSON export</a>
              </div>
              <div className="grid lg:grid-cols-3 gap-4">
                <div>
                  <div className="font-semibold mb-1">Binary (Healthy vs Unhealthy)</div>
                  {res.topk_binary.map((b,i)=>(<Bar key={i} item={b}/>))}
                </div>
                <div>
                  <div className="font-semibold mb-1">Image Top-3</div>
                  {res.topk_image.map((b,i)=>(<Bar key={i} item={b}/>))}
                </div>
                <div>
                  <div className="font-semibold mb-1">Fused Top-3</div>
                  {res.topk_fused.map((b,i)=>(<Bar key={i} item={b}/>))}
                </div>
              </div>
              <details className="mt-3"><summary className="cursor-pointer font-semibold">Raw JSON</summary>
                <pre className="mt-2 bg-black/5 dark:bg-white/5 rounded-lg p-3 text-sm overflow-x-auto">{JSON.stringify(res, null, 2)}</pre>
              </details>
            </Card>
          )}
        </motion.div>
      </div>
    </div>
  );
}
