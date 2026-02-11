import { create } from 'zustand';
import type { ModelInfo } from '../types/model';

type ModelStatus = 'idle' | 'loading' | 'loaded' | 'error';

interface ModelState {
  modelId: string | null;
  modelName: string | null;
  metadata: ModelInfo | null;
  status: ModelStatus;
  setLoading: (modelName: string) => void;
  setModel: (modelId: string, modelName: string, metadata: ModelInfo) => void;
  setError: () => void;
  clearModel: () => void;
}

export const useModelStore = create<ModelState>((set) => ({
  modelId: null,
  modelName: null,
  metadata: null,
  status: 'idle',
  setLoading: (modelName) => set({ status: 'loading', modelName }),
  setModel: (modelId, modelName, metadata) =>
    set({ modelId, modelName, metadata, status: 'loaded' }),
  setError: () => set({ status: 'error' }),
  clearModel: () =>
    set({ modelId: null, modelName: null, metadata: null, status: 'idle' }),
}));
