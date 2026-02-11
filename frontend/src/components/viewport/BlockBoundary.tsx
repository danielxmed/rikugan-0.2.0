import { useRef } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { useViewportStore } from '../../stores/viewportStore';
import { useActivationStore } from '../../stores/activationStore';
import { BLOCK_W, BLOCK_H, BLOCK_D, BLOCK_GAP, blockDimensions } from './blockLayout';

const MARGIN = 0.3;

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
    const halfD = bd / 2 + MARGIN;

    let inside = false;
    let hitLayer = -1;

    for (let i = 0; i < numLayers; i++) {
      const cy = i * (bh + BLOCK_GAP);
      const halfH = bh / 2 + MARGIN;

      if (
        camera.position.x > -halfW && camera.position.x < halfW &&
        camera.position.y > cy - halfH && camera.position.y < cy + halfH &&
        camera.position.z > -halfD && camera.position.z < halfD
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
    }
  });

  return null;
}
