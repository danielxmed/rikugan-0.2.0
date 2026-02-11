import { useRef } from 'react';
import { Text } from '@react-three/drei';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { useActivationStore } from '../../stores/activationStore';
import { BLOCK_D, blockDimensions, animatedBlockPositions } from './blockLayout';

export default function LayerLabels() {
  const numLayers = useActivationStore((s) => s.numLayers);
  const sliceMeta = useActivationStore((s) => s.sliceMeta);
  const groupRefs = useRef<(THREE.Group | null)[]>([]);

  if (numLayers === 0) return null;

  const hasData = sliceMeta && sliceMeta.seq_len > 0 && sliceMeta.d_model > 0;
  const dims = hasData
    ? blockDimensions(sliceMeta.seq_len, sliceMeta.d_model)
    : null;
  const bd = dims ? dims.depth : BLOCK_D;

  const labels = [];
  for (let i = 0; i < numLayers; i++) {
    labels.push(
      <group
        key={i}
        ref={(el: THREE.Group | null) => { groupRefs.current[i] = el; }}
      >
        <Text
          fontSize={0.14}
          color="#808090"
          anchorX="center"
          anchorY="middle"
        >
          {`L${i}`}
        </Text>
      </group>,
    );
  }

  return (
    <>
      {labels}
      <LabelAnimator numLayers={numLayers} bd={bd} groupRefs={groupRefs} />
    </>
  );
}

/** Separate component to use useFrame without conditional hook issues */
function LabelAnimator({ numLayers, bd, groupRefs }: {
  numLayers: number;
  bd: number;
  groupRefs: React.MutableRefObject<(THREE.Group | null)[]>;
}) {
  useFrame(() => {
    for (let i = 0; i < numLayers; i++) {
      const group = groupRefs.current[i];
      if (!group) continue;

      const blockPos = animatedBlockPositions[i];
      if (blockPos) {
        group.position.set(blockPos.x, blockPos.y, blockPos.z + bd / 2 + 0.15);
      }
    }
  });

  return null;
}
