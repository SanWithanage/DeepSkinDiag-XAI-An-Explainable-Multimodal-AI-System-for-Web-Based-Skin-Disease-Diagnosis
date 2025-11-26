export const API = {
  base: import.meta.env.VITE_API_BASE || "", // empty => Vite proxy to 127.0.0.1:8000
  async health() {
    const r = await fetch(`${this.base}/api/health`);
    if (!r.ok) throw new Error(await r.text());
    return r.json();
  },
  async analyze(payload: {
    file: File;
    prioritize_healthy: boolean;
    healthy_threshold: number;
    swap_binary_order: boolean;
    symptom_text?: string;
  }) {
    const fd = new FormData();
    fd.append("image", payload.file);
    fd.append("prioritize_healthy", String(payload.prioritize_healthy));
    fd.append("healthy_threshold", String(payload.healthy_threshold));
    fd.append("swap_binary_order", String(payload.swap_binary_order));
    if (payload.symptom_text) fd.append("symptom_text", payload.symptom_text);
    const r = await fetch(`${this.base}/api/analyze`, { method: "POST", body: fd });
    if (!r.ok) throw new Error(await r.text());
    return r.json();
  },
};
