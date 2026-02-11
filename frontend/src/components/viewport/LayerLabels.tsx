import { Text } from '@react-three/drei';
import { useActivationStore } from '../../stores/activationStore';
import { BLOCK_GAP, BLOCK_H, BLOCK_D, blockDimensions } from './blockLayout';

export default function LayerLabels() {
  const numLayers = useActivationStore((s) => s.numLayers);
  const sliceMeta = useActivationStore((s) => s.sliceMeta);

  if (numLayers === 0) return null;

  const hasData = sliceMeta && sliceMeta.seq_len > 0 && sliceMeta.d_model > 0;
  const dims = hasData
    ? blockDimensions(sliceMeta.seq_len, sliceMeta.d_model)
    : null;
  const bh = dims ? dims.height : BLOCK_H;
  const bd = dims ? dims.depth : BLOCK_D;

  // Position labels at the front-left edge of the block, offset slightly in Z
  // so they're visible from the default camera angle
  const labelX = 0;
  const labelZ = bd / 2 + 0.15;

  const labels = [];
  for (let i = 0; i < numLayers; i++) {
    labels.push(
      <Text
        key={i}
        position={[labelX, i * (bh + BLOCK_GAP), labelZ]}
        fontSize={0.14}
        color="#808090"
        anchorX="center"
        anchorY="middle"
      >
        {`L${i}`}
      </Text>,
    );
  }

  return <>{labels}</>;
}
