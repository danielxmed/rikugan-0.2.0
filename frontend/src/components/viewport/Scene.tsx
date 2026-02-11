import { useRef } from 'react';
import { OrbitControls, GizmoHelper, GizmoViewport } from '@react-three/drei';
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib';
import ModelBlocks from './ModelBlocks';
import BlockBoundary from './BlockBoundary';
import LayerLabels from './LayerLabels';
import SlicePlane from './SlicePlane';
import { useArrowPan } from './useArrowPan';

export default function Scene() {
  const controlsRef = useRef<OrbitControlsImpl>(null);
  useArrowPan(controlsRef);

  return (
    <>
      <color attach="background" args={['#0a0a0f']} />
      <ambientLight intensity={0.3} />
      <directionalLight position={[5, 10, 5]} intensity={0.8} />
      <gridHelper args={[20, 20, '#1a1a2e', '#1a1a2e']} />
      <ModelBlocks />
      <SlicePlane />
      <BlockBoundary />
      <LayerLabels />
      <OrbitControls ref={controlsRef} makeDefault enableDamping dampingFactor={0.1} />
      <GizmoHelper alignment="bottom-right" margin={[60, 60]}>
        <GizmoViewport />
      </GizmoHelper>
    </>
  );
}
