import { useRef, useEffect } from 'react';
import * as THREE from 'three';
import { useActivationStore, SLICE_INDEX, NUM_SLICES, getSliceOffset, getTokenProjSlice, getDimProjSlice } from '../../stores/activationStore';
import { useViewportStore } from '../../stores/viewportStore';
import { createSliceMaterial, createBandMaterial } from './heatmapShader';
import { BLOCK_W, BLOCK_H, BLOCK_D, BLOCK_GAP, blockDimensions } from './blockLayout';

// Heat palette: deep blue -> dark blue -> red -> bright red -> white-hot
const PALETTE = [
  { t: 0.0, color: new THREE.Color('#0a0a3a') },
  { t: 0.3, color: new THREE.Color('#1a1a6e') },
  { t: 0.6, color: new THREE.Color('#cc2200') },
  { t: 0.9, color: new THREE.Color('#ff4400') },
  { t: 1.0, color: new THREE.Color('#ffccaa') },
];

function samplePalette(t: number, target: THREE.Color): void {
  const clamped = Math.max(0, Math.min(1, t));
  for (let i = 0; i < PALETTE.length - 1; i++) {
    if (clamped <= PALETTE[i + 1].t) {
      const segStart = PALETTE[i];
      const segEnd = PALETTE[i + 1];
      const frac = (clamped - segStart.t) / (segEnd.t - segStart.t);
      target.lerpColors(segStart.color, segEnd.color, frac);
      return;
    }
  }
  target.copy(PALETTE[PALETTE.length - 1].color);
}

const _dummy = new THREE.Object3D();
const _color = new THREE.Color();

/** Fallback: flat-color InstancedMesh (Phase 1 behavior) */
function FallbackBlocks({ numLayers, blockHeat, gamma }: {
  numLayers: number;
  blockHeat: Float32Array;
  gamma: number;
}) {
  const meshRef = useRef<THREE.InstancedMesh>(null);

  useEffect(() => {
    if (!meshRef.current || numLayers === 0) return;
    for (let i = 0; i < numLayers; i++) {
      _dummy.position.set(0, i * (BLOCK_H + BLOCK_GAP), 0);
      _dummy.updateMatrix();
      meshRef.current.setMatrixAt(i, _dummy.matrix);
    }
    meshRef.current.instanceMatrix.needsUpdate = true;
  }, [numLayers]);

  useEffect(() => {
    if (!meshRef.current || !blockHeat || blockHeat.length === 0) return;
    let sum = 0;
    let sumSq = 0;
    for (let i = 0; i < blockHeat.length; i++) {
      sum += blockHeat[i];
      sumSq += blockHeat[i] * blockHeat[i];
    }
    const mean = sum / blockHeat.length;
    const variance = sumSq / blockHeat.length - mean * mean;
    const stddev = Math.sqrt(Math.max(0, variance));
    for (let i = 0; i < blockHeat.length; i++) {
      let normalized: number;
      if (stddev < 1e-8) {
        normalized = 0.5;
      } else {
        const z = (blockHeat[i] - mean) / stddev;
        normalized = Math.max(0, Math.min(1, (z + 2) / 4));
      }
      const display = Math.pow(normalized, gamma);
      samplePalette(display, _color);
      meshRef.current!.setColorAt(i, _color);
    }
    meshRef.current.instanceColor!.needsUpdate = true;
  }, [blockHeat, gamma]);

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, numLayers]}>
      <boxGeometry args={[BLOCK_W, BLOCK_H, BLOCK_D]} />
      <meshStandardMaterial />
    </instancedMesh>
  );
}

/**
 * Per-block GPU resources:
 * - topTex / bottomTex: full heatmap textures [d_model, seq_len]
 * - tokenProjTex: band texture [seq_len, 6] for front/back faces
 * - dimProjTex: band texture [d_model, 6] for left/right faces
 * - heatmapMat: shared for top/bottom
 * - tokenBandMat: for front/back
 * - dimBandMat: for left/right
 */
interface BlockGpuResources {
  topTex: THREE.DataTexture;
  bottomTex: THREE.DataTexture;
  tokenProjTex: THREE.DataTexture;
  dimProjTex: THREE.DataTexture;
  topMat: THREE.ShaderMaterial;
  bottomMat: THREE.ShaderMaterial;
  tokenBandMat: THREE.ShaderMaterial;
  dimBandMat: THREE.ShaderMaterial;
}

interface GpuResources {
  blocks: BlockGpuResources[];
  meshes: THREE.Mesh[];
  geometry: THREE.BoxGeometry;
}

/** Volumetric: 28 individual meshes with distinct face types */
function VolumetricBlocks({ numLayers, sliceData, sliceMeta, gamma, sliceDepth, tokenProj, dimProj }: {
  numLayers: number;
  sliceData: Float32Array;
  sliceMeta: { seq_len: number; d_model: number };
  gamma: number;
  sliceDepth: number | null;
  tokenProj: Float32Array | null;
  dimProj: Float32Array | null;
}) {
  const { seq_len, d_model } = sliceMeta;
  const dims = blockDimensions(seq_len, d_model);
  const groupRef = useRef<THREE.Group>(null);
  const gpuRef = useRef<GpuResources | null>(null);

  // Create all GPU resources and add meshes to group
  useEffect(() => {
    const group = groupRef.current;
    if (!group) return;

    // Dispose previous
    const prev = gpuRef.current;
    if (prev) {
      for (const mesh of prev.meshes) group.remove(mesh);
      for (const b of prev.blocks) {
        b.topTex.dispose(); b.bottomTex.dispose();
        b.tokenProjTex.dispose(); b.dimProjTex.dispose();
        b.topMat.dispose(); b.bottomMat.dispose();
        b.tokenBandMat.dispose(); b.dimBandMat.dispose();
      }
      prev.geometry.dispose();
    }

    const geometry = new THREE.BoxGeometry(dims.width, dims.height, dims.depth);
    const blocks: BlockGpuResources[] = [];
    const meshes: THREE.Mesh[] = [];

    for (let layer = 0; layer < numLayers; layer++) {
      // Top/bottom: full heatmap textures [d_model, seq_len]
      const topTex = new THREE.DataTexture(
        new Float32Array(seq_len * d_model), d_model, seq_len,
        THREE.RedFormat, THREE.FloatType,
      );
      topTex.minFilter = THREE.NearestFilter;
      topTex.magFilter = THREE.NearestFilter;
      topTex.needsUpdate = true;

      const bottomTex = new THREE.DataTexture(
        new Float32Array(seq_len * d_model), d_model, seq_len,
        THREE.RedFormat, THREE.FloatType,
      );
      bottomTex.minFilter = THREE.NearestFilter;
      bottomTex.magFilter = THREE.NearestFilter;
      bottomTex.needsUpdate = true;

      // Front/back: token projection bands [seq_len, 6]
      const tokenProjTex = new THREE.DataTexture(
        new Float32Array(seq_len * NUM_SLICES), seq_len, NUM_SLICES,
        THREE.RedFormat, THREE.FloatType,
      );
      tokenProjTex.minFilter = THREE.NearestFilter;
      tokenProjTex.magFilter = THREE.NearestFilter;
      tokenProjTex.needsUpdate = true;

      // Left/right: dimension projection bands [d_model, 6]
      const dimProjTex = new THREE.DataTexture(
        new Float32Array(d_model * NUM_SLICES), d_model, NUM_SLICES,
        THREE.RedFormat, THREE.FloatType,
      );
      dimProjTex.minFilter = THREE.NearestFilter;
      dimProjTex.magFilter = THREE.NearestFilter;
      dimProjTex.needsUpdate = true;

      const topMat = createSliceMaterial(topTex, gamma);
      const bottomMat = createSliceMaterial(bottomTex, gamma);
      const tokenBandMat = createBandMaterial(tokenProjTex, gamma);
      const dimBandMat = createBandMaterial(dimProjTex, gamma);

      blocks.push({ topTex, bottomTex, tokenProjTex, dimProjTex, topMat, bottomMat, tokenBandMat, dimBandMat });

      // BoxGeometry face order: [+X, -X, +Y, -Y, +Z, -Z]
      // +X(right)=dimBand, -X(left)=dimBand, +Y(top)=heatmap, -Y(bottom)=heatmap, +Z(front)=tokenBand, -Z(back)=tokenBand
      const mesh = new THREE.Mesh(geometry, [
        dimBandMat,    // 0: +X right
        dimBandMat,    // 1: -X left
        topMat,        // 2: +Y top (resid_post)
        bottomMat,     // 3: -Y bottom (resid_pre)
        tokenBandMat,  // 4: +Z front
        tokenBandMat,  // 5: -Z back
      ]);
      mesh.position.set(0, layer * (dims.height + BLOCK_GAP), 0);
      meshes.push(mesh);
      group.add(mesh);
    }

    gpuRef.current = { blocks, meshes, geometry };

    return () => {
      for (const mesh of meshes) group.remove(mesh);
      for (const b of blocks) {
        b.topTex.dispose(); b.bottomTex.dispose();
        b.tokenProjTex.dispose(); b.dimProjTex.dispose();
        b.topMat.dispose(); b.bottomMat.dispose();
        b.tokenBandMat.dispose(); b.dimBandMat.dispose();
      }
      geometry.dispose();
      gpuRef.current = null;
    };
  }, [numLayers, seq_len, d_model, dims.width, dims.height, dims.depth]); // eslint-disable-line react-hooks/exhaustive-deps

  // Upload slice data into top/bottom textures
  useEffect(() => {
    const gpu = gpuRef.current;
    if (!gpu || !sliceData || sliceData.length === 0) return;
    const sliceSize = seq_len * d_model;
    for (let layer = 0; layer < numLayers; layer++) {
      // Top = resid_post (slice index 5)
      const topOffset = getSliceOffset(layer, SLICE_INDEX.resid_post, seq_len, d_model);
      const topData = gpu.blocks[layer].topTex.image.data as Float32Array;
      topData.set(sliceData.subarray(topOffset, topOffset + sliceSize));
      gpu.blocks[layer].topTex.needsUpdate = true;

      // Bottom = resid_pre (slice index 0)
      const bottomOffset = getSliceOffset(layer, SLICE_INDEX.resid_pre, seq_len, d_model);
      const bottomData = gpu.blocks[layer].bottomTex.image.data as Float32Array;
      bottomData.set(sliceData.subarray(bottomOffset, bottomOffset + sliceSize));
      gpu.blocks[layer].bottomTex.needsUpdate = true;
    }
  }, [sliceData, numLayers, seq_len, d_model]);

  // Upload token projection data into front/back textures
  useEffect(() => {
    const gpu = gpuRef.current;
    if (!gpu || !tokenProj) return;
    for (let layer = 0; layer < numLayers; layer++) {
      const projSlice = getTokenProjSlice(tokenProj, layer, seq_len);
      const texData = gpu.blocks[layer].tokenProjTex.image.data as Float32Array;
      texData.set(projSlice);
      gpu.blocks[layer].tokenProjTex.needsUpdate = true;
    }
  }, [tokenProj, numLayers, seq_len]);

  // Upload dimension projection data into left/right textures
  useEffect(() => {
    const gpu = gpuRef.current;
    if (!gpu || !dimProj) return;
    for (let layer = 0; layer < numLayers; layer++) {
      const projSlice = getDimProjSlice(dimProj, layer, d_model);
      const texData = gpu.blocks[layer].dimProjTex.image.data as Float32Array;
      texData.set(projSlice);
      gpu.blocks[layer].dimProjTex.needsUpdate = true;
    }
  }, [dimProj, numLayers, d_model]);

  // Update gamma on all materials
  useEffect(() => {
    const gpu = gpuRef.current;
    if (!gpu) return;
    for (const b of gpu.blocks) {
      b.topMat.uniforms.uGamma.value = gamma;
      b.bottomMat.uniforms.uGamma.value = gamma;
      b.tokenBandMat.uniforms.uGamma.value = gamma;
      b.dimBandMat.uniforms.uGamma.value = gamma;
    }
  }, [gamma]);

  // Update slice indicator on band materials
  useEffect(() => {
    const gpu = gpuRef.current;
    if (!gpu) return;
    const indicator = sliceDepth !== null ? sliceDepth : -1.0;
    for (const b of gpu.blocks) {
      b.tokenBandMat.uniforms.uSliceIndicator.value = indicator;
      b.dimBandMat.uniforms.uSliceIndicator.value = indicator;
    }
  }, [sliceDepth]);

  return <group ref={groupRef} />;
}

export default function ModelBlocks() {
  const blockHeat = useActivationStore((s) => s.blockHeat);
  const numLayers = useActivationStore((s) => s.numLayers);
  const sliceData = useActivationStore((s) => s.sliceData);
  const sliceMeta = useActivationStore((s) => s.sliceMeta);
  const tokenProj = useActivationStore((s) => s.tokenProj);
  const dimProj = useActivationStore((s) => s.dimProj);
  const gamma = useViewportStore((s) => s.gamma);
  const sliceDepth = useViewportStore((s) => s.sliceDepth);

  if (numLayers === 0) return null;

  // Use volumetric view when slice data is available
  if (sliceData && sliceMeta && sliceMeta.seq_len > 0) {
    return (
      <VolumetricBlocks
        numLayers={numLayers}
        sliceData={sliceData}
        sliceMeta={sliceMeta}
        gamma={gamma}
        sliceDepth={sliceDepth}
        tokenProj={tokenProj}
        dimProj={dimProj}
      />
    );
  }

  // Fallback to flat-color InstancedMesh
  if (blockHeat) {
    return <FallbackBlocks numLayers={numLayers} blockHeat={blockHeat} gamma={gamma} />;
  }

  return null;
}
