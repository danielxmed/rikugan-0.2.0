import * as THREE from 'three';

export const BLOCK_GAP = 0.08;

// Default dimensions used by FallbackBlocks when no data is available
export const BLOCK_W = 1.2;
export const BLOCK_H = 0.3;
export const BLOCK_D = 1.2;

/**
 * Data-driven block dimensions.
 * X = seq_len (width), Y = 6 computation stages (height, fixed), Z = d_model (depth).
 * Adjust `scale` if blocks look too elongated or the stack is too tall to navigate.
 */
export function blockDimensions(seqLen: number, dModel: number): {
  width: number;
  height: number;
  depth: number;
} {
  const scale = 0.002;
  return {
    width: Math.max(0.3, seqLen * scale * 2),   // ~0.5 for 128 tokens
    height: 0.4,                                  // fixed: 6 stages of computation
    depth: Math.max(0.3, dModel * scale),         // ~2.0 for 1024 dims
  };
}

export type LayoutMode = 'stack' | 'exploded' | 'staircase';

/** Compute target positions for all blocks given layout parameters. */
export function computeBlockPositions(
  numLayers: number,
  dims: { width: number; height: number; depth: number },
  layout: LayoutMode,
  gap: number,
  step: number,
): THREE.Vector3[] {
  const positions: THREE.Vector3[] = [];
  for (let i = 0; i < numLayers; i++) {
    let x = 0, y = 0;
    switch (layout) {
      case 'stack':
        y = i * (dims.height + BLOCK_GAP);
        break;
      case 'exploded':
        y = i * (dims.height + dims.height * gap + BLOCK_GAP);
        break;
      case 'staircase':
        y = i * (dims.height + BLOCK_GAP);
        x = i * dims.width * step;
        break;
    }
    positions.push(new THREE.Vector3(x, y, 0));
  }
  return positions;
}

/**
 * Module-level shared animated positions.
 * Written by ModelBlocks in useFrame, read by SlicePlane/BlockBoundary/LayerLabels.
 */
export const animatedBlockPositions: THREE.Vector3[] = [];
