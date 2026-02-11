import { useRef, useEffect, useMemo } from 'react';
import * as THREE from 'three';
import { useActivationStore, getSliceOffset } from '../../stores/activationStore';
import { useViewportStore } from '../../stores/viewportStore';
import { createSliceMaterial } from './heatmapShader';
import { BLOCK_GAP, blockDimensions } from './blockLayout';

/** Map sliceDepth [0,1] to the nearest discrete slice index */
function depthToSliceIndex(depth: number): number {
  if (depth < 0.075) return 0;       // resid_pre
  if (depth < 0.30) return 1;        // attn_out
  if (depth < 0.475) return 2;       // delta_attn
  if (depth < 0.55) return 3;        // mlp_out
  if (depth < 0.925) return 4;       // delta_mlp
  return 5;                           // resid_post
}

interface PlaneResources {
  textures: THREE.DataTexture[];
  materials: THREE.ShaderMaterial[];
  meshes: THREE.Mesh[];
  geometry: THREE.PlaneGeometry;
}

export default function SlicePlane() {
  const sliceDepth = useViewportStore((s) => s.sliceDepth);
  const gamma = useViewportStore((s) => s.gamma);
  const numLayers = useActivationStore((s) => s.numLayers);
  const sliceData = useActivationStore((s) => s.sliceData);
  const sliceMeta = useActivationStore((s) => s.sliceMeta);

  const groupRef = useRef<THREE.Group>(null);
  const resRef = useRef<PlaneResources | null>(null);

  const hasData = sliceMeta && sliceMeta.seq_len > 0 && sliceMeta.d_model > 0;
  const seq_len = sliceMeta?.seq_len ?? 0;
  const d_model = sliceMeta?.d_model ?? 0;
  const dims = useMemo(
    () => hasData ? blockDimensions(seq_len, d_model) : null,
    [hasData, seq_len, d_model],
  );

  // Create plane meshes
  useEffect(() => {
    const group = groupRef.current;
    if (!group || !dims || numLayers === 0) return;

    // Dispose previous
    const prev = resRef.current;
    if (prev) {
      for (const mesh of prev.meshes) group.remove(mesh);
      for (const tex of prev.textures) tex.dispose();
      for (const mat of prev.materials) mat.dispose();
      prev.geometry.dispose();
    }

    const planeW = dims.width + 0.05;
    const planeD = dims.depth + 0.05;
    const geometry = new THREE.PlaneGeometry(planeW, planeD);
    const textures: THREE.DataTexture[] = [];
    const materials: THREE.ShaderMaterial[] = [];
    const meshes: THREE.Mesh[] = [];

    for (let layer = 0; layer < numLayers; layer++) {
      const tex = new THREE.DataTexture(
        new Float32Array(seq_len * d_model), d_model, seq_len,
        THREE.RedFormat, THREE.FloatType,
      );
      tex.minFilter = THREE.NearestFilter;
      tex.magFilter = THREE.NearestFilter;
      tex.needsUpdate = true;
      textures.push(tex);

      const mat = createSliceMaterial(tex, gamma);
      mat.transparent = true;
      mat.opacity = 0.85;
      mat.depthWrite = false;
      materials.push(mat);

      const mesh = new THREE.Mesh(geometry, mat);
      mesh.renderOrder = 10;
      // Rotate plane to be horizontal (XZ plane) — PlaneGeometry is in XY by default
      mesh.rotation.x = -Math.PI / 2;
      meshes.push(mesh);
      group.add(mesh);
    }

    resRef.current = { textures, materials, meshes, geometry };

    return () => {
      for (const mesh of meshes) group.remove(mesh);
      for (const tex of textures) tex.dispose();
      for (const mat of materials) mat.dispose();
      geometry.dispose();
      resRef.current = null;
    };
  }, [numLayers, seq_len, d_model, dims]);

  // Position planes and upload correct slice texture based on depth
  useEffect(() => {
    const res = resRef.current;
    if (!res || !dims || sliceDepth === null || !sliceData || sliceData.length === 0) return;

    const sliceIdx = depthToSliceIndex(sliceDepth);
    const sliceSize = seq_len * d_model;

    for (let layer = 0; layer < numLayers; layer++) {
      const blockBaseY = layer * (dims.height + BLOCK_GAP);
      const planeY = blockBaseY - dims.height / 2 + sliceDepth * dims.height;
      res.meshes[layer].position.set(0, planeY, 0);
      res.meshes[layer].visible = true;

      // Upload slice data
      const offset = getSliceOffset(layer, sliceIdx, seq_len, d_model);
      const texData = res.textures[layer].image.data as Float32Array;
      texData.set(sliceData.subarray(offset, offset + sliceSize));
      res.textures[layer].needsUpdate = true;
    }
  }, [sliceDepth, sliceData, numLayers, seq_len, d_model, dims]);

  // Hide planes when sliceDepth is null
  useEffect(() => {
    const res = resRef.current;
    if (!res) return;
    if (sliceDepth === null) {
      for (const mesh of res.meshes) mesh.visible = false;
    }
  }, [sliceDepth]);

  // Update gamma
  useEffect(() => {
    const res = resRef.current;
    if (!res) return;
    for (const mat of res.materials) {
      mat.uniforms.uGamma.value = gamma;
    }
  }, [gamma]);

  if (!hasData || numLayers === 0) return null;

  return <group ref={groupRef} />;
}
