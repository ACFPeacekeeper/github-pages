'use client';

import React, { useCallback, useEffect, useRef, useState } from 'react';
import { VISUAL_EXPERIENCE } from '../../../../configs/visualExperience';
import { useReducedMotion } from '../../../../hooks/useReducedMotion';
import { Loader2, ChevronLeft, ChevronRight } from 'lucide-react';

type ThreeModule = typeof import('three');

export interface Hotspot {
  id: string;
  yaw: number; // azimuth in radians (-PI to PI)
  pitch: number; // elevation in radians (-PI/2 to PI/2)
  label: string;
}

export interface PanoramaViewerProps {
  url: string;
  fallbackUrl?: string;
  alt: string;
  hotspots?: Hotspot[];
  onLoad?: () => void;
  onError?: (error: Error) => void;
}

export default function PanoramaViewer({
  url,
  fallbackUrl,
  alt,
  hotspots = [],
  onLoad,
  onError,
}: PanoramaViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  const [supported, setSupported] = useState(true);
  const [loading, setLoading] = useState(true);
  const [progress, setProgress] = useState(0);
  const [activeHotspotIndex, setActiveHotspotIndex] = useState<number>(-1);
  const [hotspotCoords, setHotspotCoords] = useState<Record<string, { x: number; y: number; visible: boolean }>>({});
  const [currentYaw, setCurrentYaw] = useState(0);
  
  const reducedMotion = useReducedMotion();
  
  const sceneRef = useRef<import('three').Scene | null>(null);
  const cameraRef = useRef<import('three').PerspectiveCamera | null>(null);
  const rendererRef = useRef<import('three').WebGLRenderer | null>(null);
  
  // View state
  const viewStateRef = useRef({ yaw: 0, pitch: 0 });
  const controlsStateRef = useRef<{ isDragging: boolean; lastX: number; lastY: number }>({ isDragging: false, lastX: 0, lastY: 0 });
  
  const hotspotsRef = useRef(hotspots);
  useEffect(() => {
    hotspotsRef.current = hotspots;
  }, [hotspots]);

  const updateCamera = useCallback((yaw: number, pitch: number, camera: import('three').PerspectiveCamera) => {
    // Constrain pitch to avoid flipping (-85 to 85 degrees)
    const maxPitch = (85 * Math.PI) / 180;
    const clampedPitch = Math.max(-maxPitch, Math.min(maxPitch, pitch));
    viewStateRef.current = { yaw, pitch: clampedPitch };
    setCurrentYaw(yaw);
    
    // Convert to target vector
    const phi = Math.PI / 2 - clampedPitch;
    const theta = yaw;
    
    const target = {
      x: Math.sin(phi) * Math.sin(theta),
      y: Math.cos(phi),
      z: Math.sin(phi) * Math.cos(theta)
    };
    
    camera.lookAt(target.x, target.y, target.z);
  }, []);

  const goToHotspot = useCallback((index: number) => {
    if (index >= 0 && index < hotspots.length) {
      setActiveHotspotIndex(index);
      const hs = hotspots[index];
      if (cameraRef.current) {
        updateCamera(hs.yaw, hs.pitch, cameraRef.current);
      }
    }
  }, [hotspots, updateCamera]);

  const handleNextHotspot = useCallback(() => {
    if (hotspots.length > 0) {
      goToHotspot((activeHotspotIndex + 1) % hotspots.length);
    }
  }, [activeHotspotIndex, hotspots.length, goToHotspot]);

  const handlePrevHotspot = useCallback(() => {
    if (hotspots.length > 0) {
      goToHotspot((activeHotspotIndex - 1 + hotspots.length) % hotspots.length);
    }
  }, [activeHotspotIndex, hotspots.length, goToHotspot]);

  useEffect(() => {
    const canvasElement = canvasRef.current;
    if (!canvasElement) return;
    
    let disposed = false;
    let frame = 0;
    
    const handleContextLost = (e: Event) => {
      e.preventDefault();
      console.warn('WebGL context lost in PanoramaViewer');
    };
    
    const handleContextRestored = () => {
      console.log('WebGL context restored in PanoramaViewer');
    };

    canvasElement.addEventListener('webglcontextlost', handleContextLost, false);
    canvasElement.addEventListener('webglcontextrestored', handleContextRestored, false);

    let renderer: import('three').WebGLRenderer | undefined;
    let observer: ResizeObserver | undefined;
    let intersectionObserver: IntersectionObserver | undefined;
    let isVisible = false;
    
    let THREE: ThreeModule;
    
    async function mount() {
      try {
        THREE = await import('three');
        if (disposed) return;
        
        renderer = new THREE.WebGLRenderer({ 
          canvas: canvasElement as HTMLCanvasElement,
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
      
      const camera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);
      cameraRef.current = camera;
      updateCamera(viewStateRef.current.yaw, viewStateRef.current.pitch, camera);

      // Sphere geometry for panorama (inverted so texture is inside)
      const geometry = new THREE.SphereGeometry(500, 60, 40);
      geometry.scale(-1, 1, 1); // invert

      const textureLoader = new THREE.TextureLoader();
      textureLoader.load(
        url,
        (texture) => {
          if (disposed) return;
          texture.colorSpace = THREE.SRGBColorSpace;
          const material = new THREE.MeshBasicMaterial({ map: texture });
          const mesh = new THREE.Mesh(geometry, material);
          scene.add(mesh);
          
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
            console.error('Error loading panorama:', error);
            setLoading(false);
            setSupported(false); // Fallback to flat image
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
      observer.observe(canvasElement as HTMLCanvasElement);

      const vec3 = new THREE.Vector3();
      const render = () => {
        if (disposed || !renderer) return;
        
        if (isVisible && document.visibilityState === 'visible') {
          renderer.render(scene, camera);
          
          if (hotspotsRef.current.length > 0 && containerRef.current) {
            const newCoords: Record<string, { x: number; y: number; visible: boolean }> = {};
            const containerRect = containerRef.current.getBoundingClientRect();
            
            for (const hs of hotspotsRef.current) {
              const r = 400; // inner radius to place hotspot
              const phi = Math.PI / 2 - hs.pitch;
              const theta = hs.yaw;
              
              vec3.set(
                r * Math.sin(phi) * Math.sin(theta),
                r * Math.cos(phi),
                r * Math.sin(phi) * Math.cos(theta)
              );
              
              vec3.project(camera);
              
              const x = (vec3.x * 0.5 + 0.5) * containerRect.width;
              const y = (-(vec3.y * 0.5) + 0.5) * containerRect.height;
              // Check if behind camera
              const visible = vec3.z < 1;
              
              newCoords[hs.id] = { x, y, visible };
            }
            setHotspotCoords(newCoords);
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
    }, { rootMargin: '100px' });
    
    intersectionObserver.observe(canvasElement);

    return () => {
      disposed = true;
      disconnect?.();
      intersectionObserver?.disconnect();
      canvasElement.removeEventListener('webglcontextlost', handleContextLost);
      canvasElement.removeEventListener('webglcontextrestored', handleContextRestored);
      window.cancelAnimationFrame(frame);
      
      if (sceneRef.current) {
        sceneRef.current.traverse((object: any) => {
          if (object.isMesh) {
            object.geometry?.dispose();
            if (object.material) {
              if (object.material.map) object.material.map.dispose();
              object.material.dispose();
            }
          }
        });
      }
      renderer?.dispose();
    };
  }, [url, reducedMotion, updateCamera, onLoad, onError]);

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
    if (!state.isDragging || !cameraRef.current) return;
    
    const deltaX = e.clientX - state.lastX;
    const deltaY = e.clientY - state.lastY;
    
    // Sensitivity
    const yawSpeed = 0.005;
    const pitchSpeed = 0.005;
    
    const newYaw = viewStateRef.current.yaw - deltaX * yawSpeed;
    const newPitch = viewStateRef.current.pitch + deltaY * pitchSpeed;
    
    updateCamera(newYaw, newPitch, cameraRef.current);
    
    state.lastX = e.clientX;
    state.lastY = e.clientY;
  };

  const handlePointerUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
    controlsStateRef.current.isDragging = false;
    e.currentTarget.releasePointerCapture(e.pointerId);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLCanvasElement>) => {
    if (!cameraRef.current) return;
    
    const speed = 0.1;
    let newYaw = viewStateRef.current.yaw;
    let newPitch = viewStateRef.current.pitch;
    let handled = false;
    
    switch (e.key) {
      case 'ArrowUp':
        newPitch += speed;
        handled = true;
        break;
      case 'ArrowDown':
        newPitch -= speed;
        handled = true;
        break;
      case 'ArrowLeft':
        newYaw += speed;
        handled = true;
        break;
      case 'ArrowRight':
        newYaw -= speed;
        handled = true;
        break;
    }
    
    if (handled) {
      e.preventDefault();
      updateCamera(newYaw, newPitch, cameraRef.current);
    }
  };

  // Fallback to flat image if WebGL fails
  if (!supported) {
    return (
      <div className="panorama-viewer__fallback w-full overflow-hidden bg-slate-100 rounded-lg border border-slate-200">
        <img 
          src={fallbackUrl || url} 
          alt={alt} 
          className="w-full h-auto object-cover max-h-[500px]" 
        />
        <div className="p-3 text-sm text-slate-600 bg-white border-t border-slate-200">
          360° view is not supported on this device. Displaying flat fallback.
        </div>
      </div>
    );
  }

  // Normalize yaw for minimap (0 to 1)
  const normalizedYaw = ((currentYaw % (Math.PI * 2)) + Math.PI * 2) % (Math.PI * 2) / (Math.PI * 2);

  return (
    <div ref={containerRef} className="panorama-viewer relative w-full h-full min-h-[400px] overflow-hidden bg-slate-900 rounded-lg">
      <canvas
        ref={canvasRef}
        className="block w-full h-full outline-none touch-none cursor-grab active:cursor-grabbing"
        tabIndex={0}
        aria-label={alt}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerCancel={handlePointerUp}
        onKeyDown={handleKeyDown}
      />
      
      {loading && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-slate-900/80 text-white" aria-live="polite">
          <Loader2 className="animate-spin mb-2" size={32} />
          <div className="text-sm font-medium">Loading Panorama {progress}%</div>
        </div>
      )}

      {/* Hotspots */}
      {!loading && hotspots.map((hs, i) => {
        const coords = hotspotCoords[hs.id];
        if (!coords || !coords.visible) return null;
        const isActive = activeHotspotIndex === i;
        
        return (
          <button
            key={hs.id}
            onClick={() => goToHotspot(i)}
            className={`absolute transform -translate-x-1/2 -translate-y-1/2 px-2 py-1 rounded text-xs font-medium shadow-md transition-colors ${
              isActive ? 'bg-indigo-600 text-white' : 'bg-white/90 text-slate-900 hover:bg-slate-100'
            }`}
            style={{ left: `${coords.x}px`, top: `${coords.y}px` }}
            aria-label={`Hotspot: ${hs.label}`}
            aria-pressed={isActive}
          >
            {hs.label}
          </button>
        );
      })}

      {/* Controls and Minimap UI */}
      {!loading && (
        <div className="absolute bottom-4 left-4 right-4 flex items-center justify-between gap-4 pointer-events-none">
          {hotspots.length > 0 ? (
            <div className="flex items-center gap-2 bg-slate-900/60 p-1.5 rounded-lg backdrop-blur-sm pointer-events-auto shadow-sm">
              <button 
                onClick={handlePrevHotspot}
                className="p-1 text-white hover:bg-white/20 rounded transition-colors"
                aria-label="Previous hotspot"
              >
                <ChevronLeft size={20} />
              </button>
              <span className="text-xs font-medium text-white px-2">
                {activeHotspotIndex >= 0 ? `${activeHotspotIndex + 1} / ${hotspots.length}` : 'Hotspots'}
              </span>
              <button 
                onClick={handleNextHotspot}
                className="p-1 text-white hover:bg-white/20 rounded transition-colors"
                aria-label="Next hotspot"
              >
                <ChevronRight size={20} />
              </button>
            </div>
          ) : <div />}

          {/* Minimap (yaw indicator) */}
          <div className="w-32 h-2 bg-white/20 rounded-full overflow-hidden relative pointer-events-none shadow-sm" aria-hidden="true">
            <div 
              className="absolute top-0 bottom-0 w-8 bg-white/80 rounded-full transform -translate-x-1/2 transition-transform duration-100"
              style={{ left: `${normalizedYaw * 100}%` }}
            />
          </div>
        </div>
      )}
    </div>
  );
}
