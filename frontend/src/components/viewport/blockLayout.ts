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
