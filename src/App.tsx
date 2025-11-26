import React, { useEffect, useMemo, useState } from 'react'
import { API_URL, analyzeImage, fileUrl, type AnalyzeResponse } from './lib/api'
import { Button } from './components/ui/Button'
import { Card } from './components/ui/Card'
import Dropzone from './components/Dropzone'
import { ConfidenceBars } from './components/ConfidenceBars'
import { Table } from './components/Table'
import { GuidancePanel } from './components/GuidancePanel'
import TopKChips from './components/TopKChips'
import ThemeToggle from './components/ThemeToggle'
import { buildGuidanceHTML } from './lib/guidance'   // <-- static import (no await)

function useInstallPrompt(){
  const [prompt, setPrompt] = useState<any>(null)
  useEffect(()=>{ const h=(e:any)=>{ e.preventDefault(); setPrompt(e) }; window.addEventListener('beforeinstallprompt', h); return ()=>window.removeEventListener('beforeinstallprompt', h) },[])
  const install = async ()=>{ if(!prompt) return; prompt.prompt(); try{ await prompt.userChoice }catch{}; setPrompt(null) }
  return { canInstall: !!prompt, install }
}

async function compressImage(file: File, maxSide = 1280, quality = 0.9): Promise<File> {
  try {
    const dataUrl: string = await new Promise((resolve, reject)=>{ const r=new FileReader(); r.onload=()=>resolve(r.result as string); r.onerror=reject; r.readAsDataURL(file) })
    const img: HTMLImageElement = await new Promise((resolve, reject)=>{ const i=new Image(); i.onload=()=>resolve(i); i.onerror=reject; i.src=dataUrl })
    const ratio = img.width / img.height
    let w = img.width, h = img.height
    if (Math.max(w,h) > maxSide) { if (w > h) { w = maxSide; h = Math.round(maxSide/ratio) } else { h = maxSide; w = Math.round(maxSide*ratio) } }
    const canvas = document.createElement('canvas'); canvas.width = w; canvas.height = h
    const ctx = canvas.getContext('2d')!; ctx.drawImage(img, 0, 0, w, h)
    const blob: Blob | null = await new Promise(res => canvas.toBlob(res, 'image/jpeg', quality))
    if (!blob) return file
    return new File([blob], file.name.replace(/\.[^.]+$/, '') + '.jpg', { type: 'image/jpeg' })
  } catch { return file }
}

export default function App(){
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [symptom, setSymptom] = useState('')
  const [prioritizeHealthy, setPrioritizeHealthy] = useState(true)
  const [threshold, setThreshold] = useState(0.55)
  const [swapBinary, setSwapBinary] = useState(false)
  const [loading, setLoading] = useState(false)
  const [res, setRes] = useState<AnalyzeResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [topIndex, setTopIndex] = useState(0)
  const [showGuidance, setShowGuidance] = useState(false)
  const [showTop3, setShowTop3] = useState(false)
  const { canInstall, install } = useInstallPrompt()

  useEffect(()=>{ try{ const s=JSON.parse(localStorage.getItem('skinai-settings')||'{}'); if(typeof s.prioritizeHealthy==='boolean') setPrioritizeHealthy(s.prioritizeHealthy); if(typeof s.threshold==='number') setThreshold(s.threshold); if(typeof s.swapBinary==='boolean') setSwapBinary(s.swapBinary) }catch{} },[])
  useEffect(()=>{ try{ localStorage.setItem('skinai-settings', JSON.stringify({prioritizeHealthy, threshold, swapBinary})) }catch{} },[prioritizeHealthy, threshold, swapBinary])

  function onPick(f: File){ setFile(f); setRes(null); setError(null); setTopIndex(0); setShowGuidance(false); setShowTop3(false); setPreview(URL.createObjectURL(f)) }

  async function onAnalyze(){
    if (!file) { setError('Please select an image.'); return }
    setLoading(true); setError(null); setRes(null)
    try {
      const small = await compressImage(file)
      const r = await analyzeImage(small, { prioritize_healthy: prioritizeHealthy, healthy_threshold: threshold, swap_binary_order: swapBinary, symptom_text: symptom || undefined })
      setRes(r)
      setShowGuidance(!r.low_conf_gate_triggered)
    } catch (e:any) {
      setError(e?.message || 'Analyze failed')
      if (String(e).includes('TypeError') || String(e).includes('CORS')) {
        setError(`${e?.message || 'Failed to fetch'}\nTip: If you're on a phone or different origin, enable CORS on backend and set VITE_API_URL to your laptop IP (now ${API_URL}).`)
      }
    } finally { setLoading(false) }
  }

  const finalIsHealthy = useMemo(()=> res?.final_label?.toLowerCase() === 'healthy', [res])
  const topChoices = useMemo(()=> res? res.topk_image.slice(0,3).map((t,i)=>`Top-${i+1} — ${t.label} (${t.prob.toFixed(2)})`):[], [res])
  const selectedTop = res?.topk_image?.[topIndex]

  // ✅ Synchronous guidance generation (no async/await)
  const westernHTML = useMemo(() => {
    if (!res || !selectedTop) return ''
    return buildGuidanceHTML('western', {
      topLabel: selectedTop.label,
      topConf: selectedTop.prob,
      healthyProb: res.healthy_prob,
      unhealthyProb: res.unhealthy_prob,
      healthyThreshold: res.params_echo?.HEALTHY_MIN_CONFIDENCE ?? 0.55,
    })
  }, [res, selectedTop])

  const ayurHTML = useMemo(() => {
    if (!res || !selectedTop) return ''
    return buildGuidanceHTML('ayur', {
      topLabel: selectedTop.label,
      topConf: selectedTop.prob,
      healthyProb: res.healthy_prob,
      unhealthyProb: res.unhealthy_prob,
      healthyThreshold: res.params_echo?.HEALTHY_MIN_CONFIDENCE ?? 0.55,
    })
  }, [res, selectedTop])

  return (
    <div className="max-w-6xl mx-auto px-4 py-6 space-y-4">
      <header className="flex items-start justify-between gap-3">
        <div>
          <h1 className="text-2xl font-extrabold">SkinAI — Image Diagnosis (with Healthy Override)</h1>
          <p className="text-gray-600">Upload a skin photo. The app predicts "Healthy" when the binary model is confident.</p>
        </div>
        <div className="flex gap-2 items-center">
          {canInstall && <Button onClick={install}>Install app</Button>}
          <ThemeToggle />
        </div>
      </header>

      <div className="grid md:grid-cols-3 gap-4">
        <Card className="md:col-span-1 space-y-3">
          <div className="space-y-2">
            <label className="font-semibold">Image</label>
            <Dropzone onFile={onPick} previewUrl={preview} />
          </div>
          <div className="space-y-2">
            <label className="font-semibold">(Optional) symptom text</label>
            <textarea className="w-full border rounded-xl p-2" rows={3} placeholder="e.g., mild itch, no fever …" value={symptom} onChange={(e)=>setSymptom(e.target.value)} />
          </div>
          <div className="space-y-2">
            <label className="font-semibold">Settings</label>
            <div className="flex items-center gap-2"><input type="checkbox" checked={prioritizeHealthy} onChange={(e)=>setPrioritizeHealthy(e.target.checked)} /><span>Prioritize 'Healthy' when binary is confident</span></div>
            <div className="flex items-center gap-2"><input type="checkbox" checked={swapBinary} onChange={(e)=>setSwapBinary(e.target.checked)} /><span>Swap Healthy/Unhealthy order</span></div>
            <div className="space-y-1">
              <div className="text-sm text-gray-600">Healthy override threshold: {threshold.toFixed(2)}</div>
              <input type="range" min={0.5} max={0.9} step={0.01} value={threshold} onChange={(e)=>setThreshold(parseFloat(e.target.value))} className="w-full" />
            </div>
          </div>
          <div className="flex gap-2 items-center">
            <Button onClick={onAnalyze} disabled={loading || !file}>{loading? 'Analyzing…':'Analyze'}</Button>
            <Button onClick={()=>{ setFile(null); setPreview(null); setRes(null); setError(null); setShowGuidance(false); setShowTop3(false); setTopIndex(0) }}>Clear</Button>
          </div>
          {loading && <div className="w-full h-1 rounded bg-gray-200 overflow-hidden"><div className="h-full w-2/5 bg-brand animate-pulse" /></div>}
          {error && <div className="text-red-600 text-sm whitespace-pre-wrap">{error}</div>}
          <div className="text-xs text-gray-500">Backend: {API_URL}</div>
        </Card>

        <div className="md:col-span-2 space-y-4">
          {res && (
            <>
              <Card className="space-y-3">
                <div><span className={`badge ${res.final_label.toLowerCase()==='healthy'? 'badge-healthy':'badge-unhealthy'}`}>{res.final_label.replace('_',' ')}</span></div>
                <pre className="whitespace-pre-wrap text-sm text-gray-700">{res.reason}</pre>

                {res.final_label.toLowerCase()==='healthy' && res.healthy_override && (
                  <div className="card bg-green-50 border-green-500">
                    <div className="font-bold">✅ You're likely okay</div>
                    <div>If anything changes (pain, fever, bleeding), re-run or consult a clinician.</div>
                    <div><Button className="mt-2" onClick={()=>setShowTop3(true)}>Show other possibilities (Top-3)</Button></div>
                  </div>
                )}

                {res.final_label.toLowerCase()!=='healthy' && res.low_conf_gate_triggered && (
                  <div className="card bg-indigo-50 border-indigo-500">
                    <div className="font-bold">We’re not fully certain</div>
                    <div>Lighting, focus, or framing can reduce accuracy. Try a close, well-lit photo in daylight and re-run.</div>
                    <div><Button className="mt-2" onClick={()=>setShowGuidance(true)}>Show guidance anyway</Button></div>
                  </div>
                )}
              </Card>

              {( (!res.low_conf_gate_triggered || showGuidance) || (res.final_label.toLowerCase()==='healthy' && showTop3) ) && (
                <Card className="space-y-3">
                  <div className="font-semibold">View guidance for prediction</div>
                  <TopKChips options={topChoices} valueIndex={topIndex} onChange={setTopIndex} />
                  <div className="grid md:grid-cols-2 gap-3">
                    <GuidancePanel title="Western (English)" html={westernHTML} />
                    <GuidancePanel title="Sinhala Ayurvedic (English)" html={ayurHTML} />
                  </div>
                </Card>
              )}

              <div className="grid md:grid-cols-3 gap-4">
                <Card><ConfidenceBars title="Binary (Healthy vs Unhealthy)" data={res.topk_binary} /></Card>
                <Card><ConfidenceBars title="Image Top-3" data={res.topk_image} /></Card>
                <Card><ConfidenceBars title="Fused Top-3" data={res.topk_fused} /></Card>
              </div>

              <Card className="flex flex-wrap items-center gap-3">
                <div className="font-semibold">Exports</div>
                <a className="underline" href={fileUrl(res.download_csv_url)} download>CSV export</a>
                <a className="underline" href={fileUrl(res.download_json_url)} download>JSON export</a>
              </Card>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
