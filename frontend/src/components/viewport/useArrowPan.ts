import { useEffect, useRef, type RefObject } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib';

const PAN_SPEED = 3.0; // units per second
const ARROW_KEYS = new Set(['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight']);

const _right = new THREE.Vector3();
const _up = new THREE.Vector3();
const _forward = new THREE.Vector3();
const _delta = new THREE.Vector3();

export function useArrowPan(controlsRef: RefObject<OrbitControlsImpl | null>) {
  const pressed = useRef<Set<string>>(new Set());
  const camera = useThree((s) => s.camera);

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (!ARROW_KEYS.has(e.key)) return;
      // Don't capture when terminal is focused
      if ((e.target as HTMLElement)?.closest?.('.xterm')) return;
      e.preventDefault();
      pressed.current.add(e.key);
    }

    function onKeyUp(e: KeyboardEvent) {
      pressed.current.delete(e.key);
    }

    function onBlur() {
      pressed.current.clear();
    }

    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    window.addEventListener('blur', onBlur);
    return () => {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
      window.removeEventListener('blur', onBlur);
    };
  }, []);

  useFrame((_, dt) => {
    if (pressed.current.size === 0) return;
    const controls = controlsRef.current;
    if (!controls) return;

    // Compute screen-plane vectors
    camera.getWorldDirection(_forward);
    _right.crossVectors(_forward, camera.up).normalize();
    _up.crossVectors(_right, _forward).normalize();

    _delta.set(0, 0, 0);
    const speed = PAN_SPEED * dt;

    if (pressed.current.has('ArrowRight')) _delta.addScaledVector(_right, speed);
    if (pressed.current.has('ArrowLeft')) _delta.addScaledVector(_right, -speed);
    if (pressed.current.has('ArrowUp')) _delta.addScaledVector(_up, speed);
    if (pressed.current.has('ArrowDown')) _delta.addScaledVector(_up, -speed);

    camera.position.add(_delta);
    controls.target.add(_delta);
  });
}
