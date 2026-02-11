import type { LoadModelResponse, ModelInfo, InferenceResponse } from '../types/model';

const BASE = '/api';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export function loadModel(name: string): Promise<LoadModelResponse> {
  return request<LoadModelResponse>(`/models/${encodeURIComponent(name)}/load`, {
    method: 'POST',
  });
}

export function getModelInfo(name: string): Promise<ModelInfo> {
  return request<ModelInfo>(`/models/${encodeURIComponent(name)}/info`);
}

export function runInference(prompt: string, maxNewTokens = 64): Promise<InferenceResponse> {
  return request<InferenceResponse>('/inference/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt, max_new_tokens: maxNewTokens }),
  });
}
