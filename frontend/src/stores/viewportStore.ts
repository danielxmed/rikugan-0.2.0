import { create } from 'zustand';

interface ViewportState {
  zoomLevel: 1 | 2 | 3;
  focusedLayer: number | null;
  gamma: number;
  sliceDepth: number | null;
  boundaryLayer: number | null;
  setZoomLevel: (level: 1 | 2 | 3) => void;
  setFocusedLayer: (layer: number | null) => void;
  setGamma: (gamma: number) => void;
  setSliceDepth: (depth: number | null) => void;
  setBoundaryLayer: (layer: number | null) => void;
}

export const useViewportStore = create<ViewportState>((set) => ({
  zoomLevel: 1,
  focusedLayer: null,
  gamma: 3.0,
  sliceDepth: null,
  boundaryLayer: null,
  setZoomLevel: (level) => set({ zoomLevel: level }),
  setFocusedLayer: (layer) => set({ focusedLayer: layer }),
  setGamma: (gamma) => set({ gamma: Math.max(0.1, Math.min(10, gamma)) }),
  setSliceDepth: (depth) => set({ sliceDepth: depth !== null ? Math.max(0, Math.min(1, depth)) : null }),
  setBoundaryLayer: (layer) => set({ boundaryLayer: layer }),
}));
