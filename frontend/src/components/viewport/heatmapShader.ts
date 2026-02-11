import * as THREE from 'three';

export const heatmapVertexShader = /* glsl */ `
#include <clipping_planes_pars_vertex>
varying vec2 vUv;
void main() {
  vUv = uv;
  vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mvPosition;
  #include <clipping_planes_vertex>
}
`;

export const heatmapFragmentShader = /* glsl */ `
#include <clipping_planes_pars_fragment>
uniform sampler2D uSliceTexture;
uniform float uGamma;
uniform float uEmissive;
uniform float uOpacity;

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
  #include <clipping_planes_fragment>
  float raw = texture2D(uSliceTexture, vUv).r;
  float val = pow(clamp(raw, 0.0, 1.0), uGamma);
  vec3 color = palette(val);
  // Add emissive glow for boundary hit
  color += vec3(uEmissive);
  gl_FragColor = vec4(color, uOpacity);
}
`;

export function createSliceMaterial(
  texture: THREE.DataTexture,
  gamma: number,
): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    vertexShader: heatmapVertexShader,
    fragmentShader: heatmapFragmentShader,
    clipping: true,
    uniforms: {
      uSliceTexture: { value: texture },
      uGamma: { value: gamma },
      uEmissive: { value: 0.0 },
      uOpacity: { value: 1.0 },
    },
  });
}

// --- Band Shader (for lateral projection faces) ---

export const bandVertexShader = /* glsl */ `
#include <clipping_planes_pars_vertex>
varying vec2 vUv;
void main() {
  vUv = uv;
  vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mvPosition;
  #include <clipping_planes_vertex>
}
`;

export const bandFragmentShader = /* glsl */ `
#include <clipping_planes_pars_fragment>
uniform sampler2D uBandTexture;
uniform float uGamma;
uniform float uEmissive;
uniform float uOpacity;
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
  #include <clipping_planes_fragment>
  float bandFrac = fract(vUv.y * uNumBands);

  // Dark separator between bands
  if (bandFrac < 0.04 || bandFrac > 0.96) {
    gl_FragColor = vec4(vec3(0.06), uOpacity);
    return;
  }

  // Slice indicator line
  if (uSliceIndicator >= 0.0) {
    float dist = abs(vUv.y - uSliceIndicator);
    if (dist < 0.008) {
      gl_FragColor = vec4(1.0, 1.0, 0.3, uOpacity);
      return;
    }
  }

  // Sample and colormap
  float raw = texture2D(uBandTexture, vUv).r;
  float val = pow(clamp(raw, 0.0, 1.0), uGamma);
  vec3 color = palette(val) + vec3(uEmissive);
  gl_FragColor = vec4(color, uOpacity);
}
`;

export function createBandMaterial(
  texture: THREE.DataTexture,
  gamma: number,
): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    vertexShader: bandVertexShader,
    fragmentShader: bandFragmentShader,
    clipping: true,
    uniforms: {
      uBandTexture: { value: texture },
      uGamma: { value: gamma },
      uEmissive: { value: 0.0 },
      uOpacity: { value: 1.0 },
      uNumBands: { value: 6.0 },
      uSliceIndicator: { value: -1.0 },
    },
  });
}
