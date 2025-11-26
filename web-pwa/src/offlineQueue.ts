// src/offlineQueue.ts
import { set, get, del, keys } from "idb-keyval";
import { v4 as uuid } from "uuid";

const STORE = "skinai_offline_queue";

export type PendingJob = {
  id: string;
  file: Blob;                    // photo
  filename: string;
  prioritize_healthy: boolean;
  healthy_threshold: number;
  swap_binary_order: boolean;
  symptom_text?: string;
  created_at: number;
};

export async function enqueue(job: Omit<PendingJob, "id" | "created_at">) {
  const id = uuid();
  const j: PendingJob = { id, created_at: Date.now(), ...job };
  await set(`${STORE}:${id}`, j);
  return id;
}

export async function listQueue() {
  const ks = (await keys()) as string[];
  const ids = ks.filter((k) => String(k).startsWith(`${STORE}:`));
  const jobs: PendingJob[] = [];
  for (const k of ids) jobs.push((await get(k)) as PendingJob);
  jobs.sort((a, b) => a.created_at - b.created_at);
  return jobs;
}

export async function remove(id: string) {
  await del(`${STORE}:${id}`);
}
