'use client';

import React, { useCallback, useEffect, useRef, useState } from 'react';
import { VISUAL_EXPERIENCE } from '../../configs/visualExperience';
import { useReducedMotion } from '../../hooks/useReducedMotion';
import { Loader2 } from 'lucide-react';

type ThreeModule = typeof import('three');
type GLTFLoaderModule = typeof import('three/examples/jsm/loaders/GLTFLoader.js');
type DRACOLoaderModule = typeof import('three/examples/jsm/loaders/DRACOLoader.js');

export interface CameraPreset {
  name: string;
  position: [number, number, number];
  target?: [number, number, number];
}

export interface Annotation {
  id: string;
  position: [number, number, number];
  label: string;
}

export interface ModelViewerProps {
  url: string;
  alt: string;
  cameraPresets?: CameraPreset[];
  annotations?: Annotation[];
  dracoDecoderPath?: string;
  onLoad?: () => void;
  onError?: (error: Error) => void;
}

export default function ModelViewer({
  url,
  alt,
  cameraPresets = [],
  annotations = [],
  dracoDecoderPath = 'https://www.gstatic.com/draco/versioned/decoders/1.5.7/',
  onLoad,
  onError,
}: ModelViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  const [supported, setSupported] = useState(true);
  const [loading, setLoading] = useState(true);
  const [progress, setProgress] = useState(0);
  const [activePreset, setActivePreset] = useState<string | null>(cameraPresets[0]?.name || null);
  const [annotationCoords, setAnnotationCoords] = useState<Record<string, { x: number; y: number; visible: boolean }>>({});
  
  const reducedMotion = useReducedMotion();
  
  // Refs for three.js objects needed outside mount for React state/events
  const sceneRef = useRef<import('three').Scene | null>(null);
  const cameraRef = useRef<import('three').PerspectiveCamera | null>(null);
  const rendererRef = useRef<import('three').WebGLRenderer | null>(null);
  const controlsStateRef = useRef<{ isDragging: boolean; lastX: number; lastY: number }>({ isDragging: false, lastX: 0, lastY: 0 });
  const annotationsRef = useRef(annotations);

  useEffect(() => {
    annotationsRef.current = annotations;
  }, [annotations]);

  const applyPreset = useCallback((presetName: string) => {
    const preset = cameraPresets.find(p => p.name === presetName);
    if (preset && cameraRef.current) {
      cameraRef.current.position.set(...preset.position);
      if (preset.target) {
        cameraRef.current.lookAt(...preset.target);
      } else {
        cameraRef.current.lookAt(0, 0, 0);
      }
      setActivePreset(presetName);
    }
  }, [cameraPresets]);

  useEffect(() => {
    const canvasElement = canvasRef.current;
    if (!canvasElement) return;
    
    let disposed = false;
    let frame = 0;
    
    const handleContextLost = (e: Event) => {
      e.preventDefault();
      console.warn('WebGL context lost in ModelViewer');
    };
    
    const handleContextRestored = () => {
      console.log('WebGL context restored in ModelViewer');
    };

    canvasElement.addEventListener('webglcontextlost', handleContextLost, false);
    canvasElement.addEventListener('webglcontextrestored', handleContextRestored, false);

    let renderer: import('three').WebGLRenderer | undefined;
    let observer: ResizeObserver | undefined;
    let intersectionObserver: IntersectionObserver | undefined;
    let isVisible = false;
    
    let THREE: ThreeModule;
    let GLTFLoader: GLTFLoaderModule['GLTFLoader'];
    let DRACOLoader: DRACOLoaderModule['DRACOLoader'];
    
    async function mount() {
      try {
        THREE = await import('three');
        const GLTFModule = await import('three/examples/jsm/loaders/GLTFLoader.js');
        const DRACOModule = await import('three/examples/jsm/loaders/DRACOLoader.js');
        GLTFLoader = GLTFModule.GLTFLoader;
        DRACOLoader = DRACOModule.DRACOLoader;
        
        if (disposed) return;
        
        renderer = new THREE.WebGLRenderer({ 
          canvas: canvasElement as HTMLCanvasElement, 
          alpha: true, 
          antialias: true, 
          powerPreference: 'high-performance' 
        });
        rendererRef.current = renderer;
      } catch (err) {
        setSupported(false);
        if (onError && err instanceof Error) onError(err);
        return;
      }

      const scene = new THREE.Scene();
      sceneRef.current = scene;
      
      const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000);
      if (cameraPresets.length > 0) {
        camera.position.set(...cameraPresets[0].position);
        if (cameraPresets[0].target) camera.lookAt(...cameraPresets[0].target);
      } else {
        camera.position.set(0, 0, 5);
        camera.lookAt(0, 0, 0);
      }
      cameraRef.current = camera;
      
      const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
      scene.add(ambientLight);
      const directionalLight = new THREE.DirectionalLight(0xffffff, 1.5);
      directionalLight.position.set(5, 5, 5);
      scene.add(directionalLight);

      // Load Model
      const loader = new GLTFLoader();
      const dracoLoader = new DRACOLoader();
      dracoLoader.setDecoderPath(dracoDecoderPath);
      loader.setDRACOLoader(dracoLoader);

      loader.load(
        url,
        (gltf) => {
          if (disposed) return;
          scene.add(gltf.scene);
          setLoading(false);
          if (onLoad) onLoad();
        },
        (xhr) => {
          if (xhr.lengthComputable && !disposed) {
            setProgress(Math.round((xhr.loaded / xhr.total) * 100));
          }
        },
        (error) => {
          if (!disposed) {
            console.error('Error loading model:', error);
            setLoading(false);
            if (onError && error instanceof Error) onError(error);
          }
        }
      );

      const resize = () => {
        if (!renderer) return;
        const { clientWidth, clientHeight } = canvasElement as HTMLCanvasElement;
        const dpr = Math.min(window.devicePixelRatio, reducedMotion ? VISUAL_EXPERIENCE.reducedDevicePixelRatio : VISUAL_EXPERIENCE.maxDevicePixelRatio);
        renderer.setPixelRatio(dpr);
        renderer.setSize(clientWidth, clientHeight, false);
        camera.aspect = clientWidth / Math.max(clientHeight, 1);
        camera.updateProjectionMatrix();
      };
      
      resize();
      observer = new ResizeObserver(resize);
      observer.observe(canvasElement);

      const vec3 = new THREE.Vector3();
      const render = () => {
        if (disposed || !renderer) return;
        
        if (isVisible && document.visibilityState === 'visible') {
          renderer.render(scene, camera);
          
          // Update annotation coordinates
          if (annotationsRef.current.length > 0 && containerRef.current) {
            const newCoords: Record<string, { x: number; y: number; visible: boolean }> = {};
            const containerRect = containerRef.current.getBoundingClientRect();
            
            for (const ann of annotationsRef.current) {
              vec3.set(ann.position[0], ann.position[1], ann.position[2]);
              vec3.project(camera);
              
              const x = (vec3.x * 0.5 + 0.5) * containerRect.width;
              const y = (-(vec3.y * 0.5) + 0.5) * containerRect.height;
              // Simple frustum culling
              const visible = vec3.z >= -1 && vec3.z <= 1 && vec3.x >= -1 && vec3.x <= 1 && vec3.y >= -1 && vec3.y <= 1;
              
              newCoords[ann.id] = { x, y, visible };
            }
            setAnnotationCoords(newCoords);
          }
        }
        
        frame = window.requestAnimationFrame(render);
      };
      render();

      return () => {
        observer?.disconnect();
        dracoLoader.dispose();
      };
    }

    let disconnect: (() => void) | undefined;
    
    intersectionObserver = new IntersectionObserver((entries) => {
      const entry = entries[0];
      isVisible = entry.isIntersecting;
      if (isVisible && !renderer && !disposed) {
        mount().then((cleanup) => { disconnect = cleanup; });
      }
    }, { rootMargin: '100px' });
    
    intersectionObserver.observe(canvasElement);

    return () => {
      disposed = true;
      disconnect?.();
      intersectionObserver?.disconnect();
      canvasElement.removeEventListener('webglcontextlost', handleContextLost);
      canvasElement.removeEventListener('webglcontextrestored', handleContextRestored);
      window.cancelAnimationFrame(frame);
      
      // Dispose Three.js resources
      if (sceneRef.current) {
        sceneRef.current.traverse((object: any) => {
          if (object.isMesh) {
            object.geometry?.dispose();
            if (object.material) {
              if (Array.isArray(object.material)) {
                object.material.forEach((m: any) => m.dispose());
              } else {
                object.material.dispose();
              }
            }
          }
        });
      }
      renderer?.dispose();
    };
  }, [url, dracoDecoderPath, reducedMotion, cameraPresets, onLoad, onError]);

  // Pointer events for orbiting
  const handlePointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    e.currentTarget.setPointerCapture(e.pointerId);
    controlsStateRef.current = {
      isDragging: true,
      lastX: e.clientX,
      lastY: e.clientY
    };
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const state = controlsStateRef.current;
    if (!state.isDragging || !sceneRef.current || !cameraRef.current) return;
    
    const deltaX = e.clientX - state.lastX;
    const deltaY = e.clientY - state.lastY;
    
    // Very simple orbit implementation for the scene root
    // Real implementation might use OrbitControls or rotate camera around target
    sceneRef.current.rotation.y += deltaX * 0.005;
    sceneRef.current.rotation.x += deltaY * 0.005;
    
    state.lastX = e.clientX;
    state.lastY = e.clientY;
    
    // If the user manually rotates, we might clear the active preset
    setActivePreset(null);
  };

  const handlePointerUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
    controlsStateRef.current.isDragging = false;
    e.currentTarget.releasePointerCapture(e.pointerId);
  };

  if (!supported) {
    return (
      <div className="model-viewer__fallback" role="img" aria-label={alt}>
        <div className="model-viewer__fallback-text">3D model cannot be displayed.</div>
      </div>
    );
  }

  return (
    <div ref={containerRef} className="model-viewer relative w-full h-full min-h-[300px] overflow-hidden bg-slate-900 rounded-lg">
      <canvas
        ref={canvasRef}
        className="block w-full h-full outline-none touch-none"
        tabIndex={0}
        aria-label={alt}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerCancel={handlePointerUp}
      />
      
      {loading && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-slate-900/50 text-white" aria-live="polite">
          <Loader2 className="animate-spin mb-2" size={32} />
          <div className="text-sm font-medium">Loading {progress}%</div>
        </div>
      )}

      {!loading && annotations.map((ann) => {
        const coords = annotationCoords[ann.id];
        if (!coords || !coords.visible) return null;
        return (
          <div
            key={ann.id}
            className="absolute transform -translate-x-1/2 -translate-y-1/2 bg-white/90 text-slate-900 px-2 py-1 rounded text-xs font-medium shadow-md pointer-events-auto"
            style={{ left: `${coords.x}px`, top: `${coords.y}px` }}
            aria-label={`Annotation: ${ann.label}`}
          >
            {ann.label}
          </div>
        );
      })}

      {!loading && cameraPresets.length > 0 && (
        <div className="absolute bottom-4 left-4 flex gap-2">
          {cameraPresets.map(preset => (
            <button
              key={preset.name}
              onClick={() => applyPreset(preset.name)}
              className={`px-3 py-1 text-xs font-medium rounded transition-colors ${activePreset === preset.name ? 'bg-indigo-600 text-white' : 'bg-slate-800/80 text-slate-200 hover:bg-slate-700'}`}
              aria-pressed={activePreset === preset.name}
            >
              {preset.name}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
