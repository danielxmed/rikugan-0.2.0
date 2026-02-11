import { Canvas } from '@react-three/fiber';
import Scene from './Scene';
import ViewportHUD from './ViewportHUD';

export default function Viewport() {
  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <Canvas
        camera={{ position: [0, 5, 10], fov: 50 }}
        gl={{ antialias: true, alpha: false }}
      >
        <Scene />
      </Canvas>
      <ViewportHUD />
    </div>
  );
}
