import { useRef, useEffect } from 'react';
import * as THREE from 'three';
import { useFrame, useThree } from '@react-three/fiber';
import { useActivationStore, SLICE_INDEX, NUM_SLICES, getSliceOffset, getTokenProjSlice, getDimProjSlice } from '../../stores/activationStore';
import { useViewportStore } from '../../stores/viewportStore';
import { createSliceMaterial, createBandMaterial } from './heatmapShader';
import { BLOCK_W, BLOCK_H, BLOCK_D, BLOCK_GAP, blockDimensions, computeBlockPositions, animatedBlockPositions } from './blockLayout';

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
    const dims = { width: BLOCK_W, height: BLOCK_H, depth: BLOCK_D };
    const positions = computeBlockPositions(numLayers, dims, 'stack', 1.0, 0.5);
    for (let i = 0; i < numLayers; i++) {
      _dummy.position.copy(positions[i]);
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
  clipPlanes: THREE.Plane[];  // one per block, for slice clipping
}

/** Set uOpacity uniform on all 4 materials of a block */
function setBlockOpacity(block: BlockGpuResources, opacity: number): void {
  const transparent = opacity < 0.999;
  for (const mat of [block.topMat, block.bottomMat, block.tokenBandMat, block.dimBandMat]) {
    mat.uniforms.uOpacity.value = opacity;
    mat.transparent = transparent;
  }
}

/** Set uEmissive uniform on all 4 materials of a block */
function setBlockEmissive(block: BlockGpuResources, emissive: number): void {
  for (const mat of [block.topMat, block.bottomMat, block.tokenBandMat, block.dimBandMat]) {
    mat.uniforms.uEmissive.value = emissive;
  }
}

/** Set clipping plane on all 4 materials of a block, or clear it */
function setBlockClip(block: BlockGpuResources, plane: THREE.Plane | null): void {
  const planes = plane ? [plane] : [];
  for (const mat of [block.topMat, block.bottomMat, block.tokenBandMat, block.dimBandMat]) {
    mat.clippingPlanes = planes;
  }
}

// Reusable Vector3 for target computation
const _targetVec = new THREE.Vector3();
const _cameraTarget = new THREE.Vector3();

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

  // Animation refs
  const currentPositions = useRef<THREE.Vector3[]>([]);
  const currentOpacities = useRef<number[]>([]);
  // Camera target transition: only actively lerp when layout/isolation changes
  const cameraTransitioning = useRef(false);
  const prevLayoutKey = useRef('');  // serialized layout+isolation state for change detection

  // Access controls for camera target during isolation
  const controls = useThree((s) => s.controls) as { target: THREE.Vector3 } | null;

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

    // Compute initial positions
    const initPositions = computeBlockPositions(numLayers, dims, 'stack', 1.0, 0.5);

    // Initialize animation state
    currentPositions.current = [];
    currentOpacities.current = [];
    // Initialize shared animated positions array
    animatedBlockPositions.length = 0;

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
      mesh.position.copy(initPositions[layer]);
      meshes.push(mesh);
      group.add(mesh);

      currentPositions.current.push(initPositions[layer].clone());
      currentOpacities.current.push(1.0);
      animatedBlockPositions.push(initPositions[layer].clone());
    }

    // Create one clipping plane per block (updated each frame in useFrame)
    const clipPlanes: THREE.Plane[] = [];
    for (let i = 0; i < numLayers; i++) {
      clipPlanes.push(new THREE.Plane(new THREE.Vector3(0, -1, 0), 0));
    }

    gpuRef.current = { blocks, meshes, geometry, clipPlanes };

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
      animatedBlockPositions.length = 0;
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

  // Animation loop: position, opacity, emissive, camera target
  useFrame((_state, delta) => {
    const gpu = gpuRef.current;
    if (!gpu || gpu.meshes.length === 0) return;

    const { layoutMode, layoutGap, layoutStep,
            isoLayoutMode, isoLayoutGap, isoLayoutStep,
            isolatedLayers, boundaryLayer } =
      useViewportStore.getState();

    const n = gpu.meshes.length;
    const lerpFactor = 1 - Math.pow(0.001, delta);

    // 1. Compute layout target positions
    const layoutPositions = computeBlockPositions(n, dims, layoutMode, layoutGap, layoutStep);

    // 2. Compute stack midpoint for camera target default
    const stackMidY = ((n - 1) * (dims.height + BLOCK_GAP)) / 2;

    // 3. Compute isolation override targets
    const hasIsolation = isolatedLayers.length > 0;

    // Find the rightmost X extent of the layout for isolation placement
    let maxLayoutX = 0;
    // Pre-compute iso positions (horizontal layout: X = primary axis, Y = secondary)
    let isoTargets: THREE.Vector3[] | null = null;
    if (hasIsolation) {
      for (let i = 0; i < n; i++) {
        maxLayoutX = Math.max(maxLayoutX, layoutPositions[i].x);
      }
      const isoCount = isolatedLayers.length;
      const startX = maxLayoutX + dims.width * 2.5;
      isoTargets = [];
      for (let j = 0; j < isoCount; j++) {
        let dx = 0, dy = 0;
        switch (isoLayoutMode) {
          case 'stack':
            dx = j * (dims.width + BLOCK_GAP * 2);
            break;
          case 'exploded':
            dx = j * (dims.width + dims.width * isoLayoutGap + BLOCK_GAP * 2);
            break;
          case 'staircase':
            dx = j * (dims.width + BLOCK_GAP * 2);
            dy = j * dims.height * isoLayoutStep;
            break;
        }
        isoTargets.push(new THREE.Vector3(startX + dx, stackMidY + dy, 0));
      }
    }

    for (let i = 0; i < n; i++) {
      // Determine target position
      if (hasIsolation && isolatedLayers.includes(i) && isoTargets) {
        const isolIdx = isolatedLayers.indexOf(i);
        _targetVec.copy(isoTargets[isolIdx]);
      } else {
        _targetVec.copy(layoutPositions[i]);
      }

      // Lerp position
      currentPositions.current[i].lerp(_targetVec, lerpFactor);
      gpu.meshes[i].position.copy(currentPositions.current[i]);

      // Write to shared animated positions
      if (animatedBlockPositions[i]) {
        animatedBlockPositions[i].copy(currentPositions.current[i]);
      }

      // Determine target opacity
      const targetOpacity = hasIsolation && !isolatedLayers.includes(i) ? 0.15 : 1.0;

      // Lerp opacity
      const currOp = currentOpacities.current[i];
      currentOpacities.current[i] = currOp + (targetOpacity - currOp) * lerpFactor;
      setBlockOpacity(gpu.blocks[i], currentOpacities.current[i]);

      // Emissive glow for boundary hit
      const emissiveTarget = (i === boundaryLayer) ? 0.15 : 0.0;
      setBlockEmissive(gpu.blocks[i], emissiveTarget);

      // Clipping plane for slice cut
      const currentSliceDepth = useViewportStore.getState().sliceDepth;
      if (currentSliceDepth !== null && gpu.clipPlanes[i]) {
        // Clip everything above the slice Y: normal (0,-1,0), constant = sliceY
        // Plane eq: -y + constant >= 0  =>  y <= constant
        const sliceY = currentPositions.current[i].y
          - dims.height / 2
          + currentSliceDepth * dims.height;
        gpu.clipPlanes[i].set(new THREE.Vector3(0, -1, 0), sliceY);
        setBlockClip(gpu.blocks[i], gpu.clipPlanes[i]);
      } else {
        setBlockClip(gpu.blocks[i], null);
      }
    }

    // 4. Camera target: only lerp during layout/isolation transitions
    if (controls && 'target' in controls) {
      // Detect layout/isolation state changes
      const layoutKey = `${layoutMode}:${layoutGap}:${layoutStep}:${isoLayoutMode}:${isoLayoutGap}:${isoLayoutStep}:${isolatedLayers.join(',')}`;
      if (layoutKey !== prevLayoutKey.current) {
        prevLayoutKey.current = layoutKey;
        cameraTransitioning.current = true;
      }

      if (cameraTransitioning.current) {
        if (hasIsolation) {
          // Target = centroid of isolated blocks
          _cameraTarget.set(0, 0, 0);
          for (const li of isolatedLayers) {
            _cameraTarget.add(currentPositions.current[li]);
          }
          _cameraTarget.divideScalar(isolatedLayers.length);
        } else {
          // Target = center of layout
          _cameraTarget.set(
            layoutPositions.length > 0 ? layoutPositions[Math.floor(n / 2)].x : 0,
            stackMidY,
            0,
          );
        }

        controls.target.lerp(_cameraTarget, lerpFactor);

        // Stop transitioning once converged
        const dist = controls.target.distanceTo(_cameraTarget);
        if (dist < 0.01) {
          cameraTransitioning.current = false;
        }
      }
    }
  });

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
