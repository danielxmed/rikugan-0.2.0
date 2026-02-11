import { create } from 'zustand';

export const SLICE_NAMES = [
  'resid_pre', 'attn_out', 'delta_attn', 'mlp_out', 'delta_mlp', 'resid_post',
] as const;

export const SLICE_INDEX: Record<string, number> = {
  resid_pre: 0,
  attn_out: 1,
  delta_attn: 2,
  mlp_out: 3,
  delta_mlp: 4,
  resid_post: 5,
};

export const NUM_SLICES = SLICE_NAMES.length;

export function getSliceOffset(
  layer: number,
  sliceIdx: number,
  seqLen: number,
  dModel: number,
): number {
  return (layer * NUM_SLICES + sliceIdx) * seqLen * dModel;
}

interface SliceMeta {
  seq_len: number;
  d_model: number;
}

interface ProjMeta {
  num_layers: number;
  seq_len: number;
  d_model: number;
  num_stages: number;
  token_proj_size: number;
}

interface ActivationState {
  blockHeat: Float32Array | null;
  numLayers: number;
  prompt: string | null;
  sliceData: Float32Array | null;
  sliceMeta: SliceMeta | null;
  tokenProj: Float32Array | null;
  dimProj: Float32Array | null;
  projMeta: ProjMeta | null;
  setFrame: (heat: number[], numLayers: number, prompt: string) => void;
  setSliceMeta: (meta: SliceMeta) => void;
  setSliceData: (data: Float32Array) => void;
  setTokenProj: (data: Float32Array) => void;
  setDimProj: (data: Float32Array) => void;
  setProjMeta: (meta: ProjMeta) => void;
  clear: () => void;
}

export function getTokenProjSlice(
  tokenProj: Float32Array,
  layer: number,
  seqLen: number,
): Float32Array {
  const offset = layer * NUM_SLICES * seqLen;
  return tokenProj.subarray(offset, offset + NUM_SLICES * seqLen);
}

export function getDimProjSlice(
  dimProj: Float32Array,
  layer: number,
  dModel: number,
): Float32Array {
  const offset = layer * NUM_SLICES * dModel;
  return dimProj.subarray(offset, offset + NUM_SLICES * dModel);
}

export const useActivationStore = create<ActivationState>((set) => ({
  blockHeat: null,
  numLayers: 0,
  prompt: null,
  sliceData: null,
  sliceMeta: null,
  tokenProj: null,
  dimProj: null,
  projMeta: null,
  setFrame: (heat, numLayers, prompt) =>
    set({ blockHeat: new Float32Array(heat), numLayers, prompt }),
  setSliceMeta: (meta) => set({ sliceMeta: meta }),
  setSliceData: (data) => set({ sliceData: data }),
  setTokenProj: (data) => set({ tokenProj: data }),
  setDimProj: (data) => set({ dimProj: data }),
  setProjMeta: (meta) => set({ projMeta: meta }),
  clear: () =>
    set({ blockHeat: null, numLayers: 0, prompt: null, sliceData: null, sliceMeta: null,
           tokenProj: null, dimProj: null, projMeta: null }),
}));
