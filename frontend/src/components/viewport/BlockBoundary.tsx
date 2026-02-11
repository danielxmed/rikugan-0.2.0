import { useRef } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { useViewportStore } from '../../stores/viewportStore';
import { useActivationStore } from '../../stores/activationStore';
import { BLOCK_W, BLOCK_H, BLOCK_D, BLOCK_GAP, blockDimensions, animatedBlockPositions } from './blockLayout';

const MARGIN = 0.3;
const GLOW_CLEAR_DIST = MARGIN * 3;

export default function BlockBoundary({
  onBoundaryReached,
}: {
  onBoundaryReached?: (layerIndex: number) => void;
}) {
  const lastValidPos = useRef(new THREE.Vector3(0, 5, 10));
  const { camera } = useThree();

  useFrame(() => {
    const numLayers = useActivationStore.getState().numLayers;
    if (numLayers === 0) {
      lastValidPos.current.copy(camera.position);
      return;
    }

    const sliceMeta = useActivationStore.getState().sliceMeta;
    const hasData = sliceMeta && sliceMeta.seq_len > 0 && sliceMeta.d_model > 0;
    const dims = hasData
      ? blockDimensions(sliceMeta.seq_len, sliceMeta.d_model)
      : null;
    const bw = dims ? dims.width : BLOCK_W;
    const bh = dims ? dims.height : BLOCK_H;
    const bd = dims ? dims.depth : BLOCK_D;

    const halfW = bw / 2 + MARGIN;
    const halfH = bh / 2 + MARGIN;
    const halfD = bd / 2 + MARGIN;

    const isolatedLayers = useViewportStore.getState().isolatedLayers;
    const hasIsolation = isolatedLayers.length > 0;

    let inside = false;
    let hitLayer = -1;

    for (let i = 0; i < numLayers; i++) {
      // Skip non-isolated blocks when isolation is active
      if (hasIsolation && !isolatedLayers.includes(i)) continue;

      // Use animated positions if available, otherwise fallback
      const cx = animatedBlockPositions[i]?.x ?? 0;
      const cy = animatedBlockPositions[i]?.y ?? i * (bh + BLOCK_GAP);
      const cz = animatedBlockPositions[i]?.z ?? 0;

      if (
        camera.position.x > cx - halfW && camera.position.x < cx + halfW &&
        camera.position.y > cy - halfH && camera.position.y < cy + halfH &&
        camera.position.z > cz - halfD && camera.position.z < cz + halfD
      ) {
        inside = true;
        hitLayer = i;
        break;
      }
    }

    if (inside) {
      camera.position.copy(lastValidPos.current);
      const prev = useViewportStore.getState().boundaryLayer;
      if (prev !== hitLayer) {
        useViewportStore.getState().setBoundaryLayer(hitLayer);
        onBoundaryReached?.(hitLayer);
      }
    } else {
      lastValidPos.current.copy(camera.position);

      // Clear boundary glow when camera moves away from the block
      const currentBoundary = useViewportStore.getState().boundaryLayer;
      if (currentBoundary !== null) {
        const bx = animatedBlockPositions[currentBoundary]?.x ?? 0;
        const by = animatedBlockPositions[currentBoundary]?.y ?? currentBoundary * (bh + BLOCK_GAP);
        const bz = animatedBlockPositions[currentBoundary]?.z ?? 0;
        const dx = camera.position.x - bx;
        const dy = camera.position.y - by;
        const dz = camera.position.z - bz;
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (dist > GLOW_CLEAR_DIST) {
          useViewportStore.getState().setBoundaryLayer(null);
        }
      }
    }
  });

  return null;
}
