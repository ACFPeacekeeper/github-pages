'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { Pause, Play, RotateCcw } from 'lucide-react';
import { VISUAL_EXPERIENCE } from '../../configs/visualExperience';
import { useReducedMotion } from '../../hooks/useReducedMotion';

type ThreeModule = typeof import('three');

export default function HeroModel() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rotationRef = useRef({ x: 0.35, y: -0.5 });
  const pausedRef = useRef(false);
  const dragRef = useRef<{ x: number; y: number } | null>(null);
  const [paused, setPaused] = useState(false);
  const [supported, setSupported] = useState(true);
  const reducedMotion = useReducedMotion();

  useEffect(() => {
    pausedRef.current = paused;
  }, [paused]);

  const rotate = useCallback((delta: number) => {
    rotationRef.current.y += delta;
  }, []);

  const reset = useCallback(() => {
    rotationRef.current = { x: 0.35, y: -0.5 };
  }, []);

  useEffect(() => {
    const canvasElement = canvasRef.current;
    if (!canvasElement) return;
    const targetCanvas: HTMLCanvasElement = canvasElement;
    let disposed = false;
    let frame = 0;
    
    const handleContextLost = (e: Event) => {
      e.preventDefault();
      console.warn('WebGL context lost');
    };
    
    const handleContextRestored = () => {
      console.log('WebGL context restored');
    };

    targetCanvas.addEventListener('webglcontextlost', handleContextLost, false);
    targetCanvas.addEventListener('webglcontextrestored', handleContextRestored, false);

    let renderer: import('three').WebGLRenderer | undefined;
    let geometry: import('three').BufferGeometry | undefined;
    let material: import('three').Material | undefined;
    let wireGeometry: import('three').BufferGeometry | undefined;
    let wireMaterial: import('three').Material | undefined;
    let observer: ResizeObserver | undefined;
    let intersectionObserver: IntersectionObserver | undefined;

    let frames = 0;
    let lastTime = performance.now();
    let isVisible = false;

    async function mount() {
      let THREE: ThreeModule;
      try {
        THREE = await import('three');
        if (disposed) return;
        renderer = new THREE.WebGLRenderer({ canvas: targetCanvas, alpha: true, antialias: true, powerPreference: 'high-performance' });
      } catch {
        setSupported(false);
        return;
      }

      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(34, 1, 0.1, 100);
      camera.position.set(0, 0, 6.2);
      geometry = new THREE.IcosahedronGeometry(1.65, 2);
      material = new THREE.MeshPhysicalMaterial({ color: '#7dd3fc', emissive: '#312e81', emissiveIntensity: 0.7, metalness: 0.32, roughness: 0.18, transmission: 0.14, clearcoat: 1, flatShading: true });
      const mesh = new THREE.Mesh(geometry, material);
      wireGeometry = new THREE.WireframeGeometry(geometry);
      wireMaterial = new THREE.LineBasicMaterial({ color: '#c4b5fd', transparent: true, opacity: 0.35 });
      const wireframe = new THREE.LineSegments(wireGeometry, wireMaterial);
      mesh.add(wireframe);
      scene.add(mesh);
      scene.add(new THREE.HemisphereLight('#e0f2fe', '#312e81', 2.2));
      const light = new THREE.PointLight('#f0abfc', 35, 12);
      light.position.set(3, 2, 4);
      scene.add(light);

      const resize = () => {
        if (!renderer) return;
        const { clientWidth, clientHeight } = targetCanvas;
        const dpr = Math.min(window.devicePixelRatio, reducedMotion ? VISUAL_EXPERIENCE.reducedDevicePixelRatio : VISUAL_EXPERIENCE.maxDevicePixelRatio);
        renderer.setPixelRatio(dpr);
        renderer.setSize(clientWidth, clientHeight, false);
        camera.aspect = clientWidth / Math.max(clientHeight, 1);
        camera.updateProjectionMatrix();
      };
      resize();
      observer = new ResizeObserver(resize);
      observer.observe(targetCanvas);

      const render = () => {
        if (disposed || !renderer) return;
        
        if (isVisible && document.visibilityState === 'visible') {
            mesh.rotation.x = rotationRef.current.x;
            mesh.rotation.y = rotationRef.current.y;
            if (!pausedRef.current && !reducedMotion) rotationRef.current.y += 0.0025;
            renderer.render(scene, camera);
            
            frames++;
            const now = performance.now();
            if (now - lastTime >= 1000) {
              const fps = Math.round((frames * 1000) / (now - lastTime));
              const memory = renderer.info.memory;
              console.debug(`[HeroModel] FPS: ${fps} | Geometries: ${memory.geometries} | Textures: ${memory.textures}`);
              frames = 0;
              lastTime = now;
            }
        }
        
        frame = window.requestAnimationFrame(render);
      };
      render();

      return () => {
        observer?.disconnect();
      };
    }

    let disconnect: (() => void) | undefined;
    
    intersectionObserver = new IntersectionObserver((entries) => {
        const entry = entries[0];
        isVisible = entry.isIntersecting;
        if (isVisible && !renderer && !disposed) {
            mount().then((cleanup) => { disconnect = cleanup; });
        }
    }, { rootMargin: '50px' });
    
    intersectionObserver.observe(targetCanvas);

    return () => {
      disposed = true;
      disconnect?.();
      intersectionObserver?.disconnect();
      targetCanvas.removeEventListener('webglcontextlost', handleContextLost);
      targetCanvas.removeEventListener('webglcontextrestored', handleContextRestored);
      window.cancelAnimationFrame(frame);
      geometry?.dispose();
      material?.dispose();
      wireGeometry?.dispose();
      wireMaterial?.dispose();
      renderer?.dispose();
    };
  }, [reducedMotion]);

  return (
    <div className="hero-model" onKeyDown={(event) => {
      if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
        event.preventDefault();
        rotate(event.key === 'ArrowLeft' ? -VISUAL_EXPERIENCE.modelRotationStep : VISUAL_EXPERIENCE.modelRotationStep);
      }
    }}>
      <div className="hero-model__halo" aria-hidden="true" />
      {supported ? <canvas
        ref={canvasRef}
        tabIndex={0}
        aria-label="Interactive faceted research model. Drag to inspect or use the arrow keys to rotate."
        onPointerDown={(event) => {
          dragRef.current = { x: event.clientX, y: event.clientY };
          event.currentTarget.setPointerCapture(event.pointerId);
        }}
        onPointerMove={(event) => {
          if (!dragRef.current) return;
          rotationRef.current.y += (event.clientX - dragRef.current.x) * 0.008;
          rotationRef.current.x += (event.clientY - dragRef.current.y) * 0.008;
          dragRef.current = { x: event.clientX, y: event.clientY };
        }}
        onPointerUp={(event) => {
          dragRef.current = null;
          event.currentTarget.releasePointerCapture(event.pointerId);
        }}
        onPointerCancel={() => { dragRef.current = null; }}
      /> : <div className="hero-model__fallback" role="img" aria-label="Faceted research model static preview">◇</div>}
      <div className="hero-model__label"><span /> Live research object</div>
      <div className="hero-model__controls">
        <button type="button" onClick={() => setPaused((value) => !value)} aria-label={paused ? 'Resume model rotation' : 'Pause model rotation'}>{paused ? <Play size={15} /> : <Pause size={15} />}</button>
        <button type="button" onClick={reset} aria-label="Reset model rotation"><RotateCcw size={15} /></button>
      </div>
    </div>
  );
}
