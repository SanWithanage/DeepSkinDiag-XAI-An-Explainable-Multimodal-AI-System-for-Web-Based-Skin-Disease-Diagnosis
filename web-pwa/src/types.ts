// src/types.ts
export type BarItem = { label: string; prob: number };

export type AnalyzeResponse = {
  run_id: string;
  final_label: string;
  reason: string;
  healthy_override: boolean;
  healthy_prob: number;
  unhealthy_prob: number;
  topk_binary: BarItem[];
  topk_image: BarItem[];
  topk_fused: BarItem[];
  risk_flags: string[];
  params_echo: Record<string, unknown>;
  download_csv_url: string;
  download_json_url: string;
  low_conf_gate_triggered: boolean;
};
