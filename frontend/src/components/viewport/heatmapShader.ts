import * as THREE from 'three';

export const heatmapVertexShader = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

export const heatmapFragmentShader = /* glsl */ `
uniform sampler2D uSliceTexture;
uniform float uGamma;
uniform float uEmissive;

varying vec2 vUv;

// 5-point palette: deep blue -> dark blue -> red -> bright red -> white-hot
vec3 palette(float t) {
  // Palette stops: 0.0, 0.3, 0.6, 0.9, 1.0
  vec3 c0 = vec3(0.039, 0.039, 0.227); // #0a0a3a
  vec3 c1 = vec3(0.102, 0.102, 0.431); // #1a1a6e
  vec3 c2 = vec3(0.800, 0.133, 0.000); // #cc2200
  vec3 c3 = vec3(1.000, 0.267, 0.000); // #ff4400
  vec3 c4 = vec3(1.000, 0.800, 0.667); // #ffccaa

  if (t < 0.3) return mix(c0, c1, t / 0.3);
  if (t < 0.6) return mix(c1, c2, (t - 0.3) / 0.3);
  if (t < 0.9) return mix(c2, c3, (t - 0.6) / 0.3);
  return mix(c3, c4, (t - 0.9) / 0.1);
}

void main() {
  float raw = texture2D(uSliceTexture, vUv).r;
  float val = pow(clamp(raw, 0.0, 1.0), uGamma);
  vec3 color = palette(val);
  // Add emissive glow for boundary hit
  color += vec3(uEmissive);
  gl_FragColor = vec4(color, 1.0);
}
`;

export function createSliceMaterial(
  texture: THREE.DataTexture,
  gamma: number,
): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    vertexShader: heatmapVertexShader,
    fragmentShader: heatmapFragmentShader,
    uniforms: {
      uSliceTexture: { value: texture },
      uGamma: { value: gamma },
      uEmissive: { value: 0.0 },
    },
  });
}

// --- Band Shader (for lateral projection faces) ---

export const bandVertexShader = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

export const bandFragmentShader = /* glsl */ `
uniform sampler2D uBandTexture;
uniform float uGamma;
uniform float uEmissive;
uniform float uNumBands;
uniform float uSliceIndicator;

varying vec2 vUv;

vec3 palette(float t) {
  vec3 c0 = vec3(0.039, 0.039, 0.227);
  vec3 c1 = vec3(0.102, 0.102, 0.431);
  vec3 c2 = vec3(0.800, 0.133, 0.000);
  vec3 c3 = vec3(1.000, 0.267, 0.000);
  vec3 c4 = vec3(1.000, 0.800, 0.667);

  if (t < 0.3) return mix(c0, c1, t / 0.3);
  if (t < 0.6) return mix(c1, c2, (t - 0.3) / 0.3);
  if (t < 0.9) return mix(c2, c3, (t - 0.6) / 0.3);
  return mix(c3, c4, (t - 0.9) / 0.1);
}

void main() {
  float bandFrac = fract(vUv.y * uNumBands);

  // Dark separator between bands
  if (bandFrac < 0.04 || bandFrac > 0.96) {
    gl_FragColor = vec4(vec3(0.06), 1.0);
    return;
  }

  // Slice indicator line
  if (uSliceIndicator >= 0.0) {
    float dist = abs(vUv.y - uSliceIndicator);
    if (dist < 0.008) {
      gl_FragColor = vec4(1.0, 1.0, 0.3, 1.0);
      return;
    }
  }

  // Sample and colormap
  float raw = texture2D(uBandTexture, vUv).r;
  float val = pow(clamp(raw, 0.0, 1.0), uGamma);
  vec3 color = palette(val) + vec3(uEmissive);
  gl_FragColor = vec4(color, 1.0);
}
`;

export function createBandMaterial(
  texture: THREE.DataTexture,
  gamma: number,
): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    vertexShader: bandVertexShader,
    fragmentShader: bandFragmentShader,
    uniforms: {
      uBandTexture: { value: texture },
      uGamma: { value: gamma },
      uEmissive: { value: 0.0 },
      uNumBands: { value: 6.0 },
      uSliceIndicator: { value: -1.0 },
    },
  });
}
