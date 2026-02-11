import { useActivationStore } from '../../stores/activationStore';
import { useViewportStore } from '../../stores/viewportStore';

const panelStyle: React.CSSProperties = {
  background: 'rgba(10,10,15,0.7)',
  borderRadius: 4,
  padding: '6px 10px',
  fontFamily: "'JetBrains Mono', monospace",
  fontSize: 11,
  color: '#a0a0b0',
  lineHeight: 1.6,
};

export default function ViewportHUD() {
  const numLayers = useActivationStore((s) => s.numLayers);
  const sliceDepth = useViewportStore((s) => s.sliceDepth);
  const gamma = useViewportStore((s) => s.gamma);

  if (numLayers === 0) return null;

  return (
    <div
      style={{
        position: 'absolute',
        inset: 0,
        pointerEvents: 'none',
        overflow: 'hidden',
      }}
    >
      {/* Top-left: Face Legend */}
      <div style={{ position: 'absolute', top: 8, left: 8, ...panelStyle }}>
        <div style={{ whiteSpace: 'pre' }}>{'Top / Bottom'.padEnd(16)}resid_post / resid_pre  (full heatmap)</div>
        <div style={{ whiteSpace: 'pre' }}>{'Front / Back'.padEnd(16)}token trajectory         (6 bands)</div>
        <div style={{ whiteSpace: 'pre' }}>{'Left / Right'.padEnd(16)}dimension trajectory     (6 bands)</div>
      </div>

      {/* Top-right: Slice Depth Indicator */}
      <div style={{ position: 'absolute', top: 8, right: 8, ...panelStyle }}>
        Slice: {sliceDepth !== null ? sliceDepth.toFixed(2) : 'off'}
      </div>

      {/* Bottom-left: Color Scale Bar */}
      <div style={{ position: 'absolute', bottom: 8, left: 8, ...panelStyle }}>
        <div
          style={{
            width: 120,
            height: 12,
            borderRadius: 2,
            background:
              'linear-gradient(to right, #0a0a3a 0%, #1a1a6e 30%, #cc2200 60%, #ff4400 90%, #ffccaa 100%)',
            marginBottom: 4,
          }}
        />
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span>Low</span>
          <span>High</span>
        </div>
        <div style={{ textAlign: 'center', marginTop: 2 }}>
          &gamma; {gamma.toFixed(1)}
        </div>
      </div>
    </div>
  );
}
