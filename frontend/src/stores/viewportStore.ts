import { create } from 'zustand';
import type { LayoutMode } from '../components/viewport/blockLayout';

interface ViewportState {
  zoomLevel: 1 | 2 | 3;
  focusedLayer: number | null;
  gamma: number;
  sliceDepth: number | null;
  boundaryLayer: number | null;
  layoutMode: LayoutMode;
  layoutGap: number;
  layoutStep: number;
  isoLayoutMode: LayoutMode;
  isoLayoutGap: number;
  isoLayoutStep: number;
  isolatedLayers: number[];
  setZoomLevel: (level: 1 | 2 | 3) => void;
  setFocusedLayer: (layer: number | null) => void;
  setGamma: (gamma: number) => void;
  setSliceDepth: (depth: number | null) => void;
  setBoundaryLayer: (layer: number | null) => void;
  setLayout: (mode: LayoutMode, param?: number) => void;
  setIsoLayout: (mode: LayoutMode, param?: number) => void;
  isolateLayer: (layer: number) => void;
  releaseLayer: (layer: number | null) => void;
}

export const useViewportStore = create<ViewportState>((set) => ({
  zoomLevel: 1,
  focusedLayer: null,
  gamma: 3.0,
  sliceDepth: null,
  boundaryLayer: null,
  layoutMode: 'stack',
  layoutGap: 1.0,
  layoutStep: 0.5,
  isoLayoutMode: 'stack',
  isoLayoutGap: 1.0,
  isoLayoutStep: 0.5,
  isolatedLayers: [],
  setZoomLevel: (level) => set({ zoomLevel: level }),
  setFocusedLayer: (layer) => set({ focusedLayer: layer }),
  setGamma: (gamma) => set({ gamma: Math.max(0.1, Math.min(10, gamma)) }),
  setSliceDepth: (depth) => set({ sliceDepth: depth !== null ? Math.max(0, Math.min(1, depth)) : null }),
  setBoundaryLayer: (layer) => set({ boundaryLayer: layer }),
  setLayout: (mode, param?) => set(() => {
    const update: Partial<ViewportState> = { layoutMode: mode };
    if (mode === 'exploded' && param !== undefined) update.layoutGap = param;
    if (mode === 'staircase' && param !== undefined) update.layoutStep = param;
    return update;
  }),
  setIsoLayout: (mode, param?) => set(() => {
    const update: Partial<ViewportState> = { isoLayoutMode: mode };
    if (mode === 'exploded' && param !== undefined) update.isoLayoutGap = param;
    if (mode === 'staircase' && param !== undefined) update.isoLayoutStep = param;
    return update;
  }),
  isolateLayer: (layer) => set((state) => {
    if (state.isolatedLayers.length >= 4 || state.isolatedLayers.includes(layer)) return state;
    return { isolatedLayers: [...state.isolatedLayers, layer] };
  }),
  releaseLayer: (layer) => set((state) => {
    if (layer === null) return { isolatedLayers: [] };
    return { isolatedLayers: state.isolatedLayers.filter((l) => l !== layer) };
  }),
}));
