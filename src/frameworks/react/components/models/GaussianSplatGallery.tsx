'use client';
import React, { useState, useEffect, useRef } from 'react';

/**
 * IF11: Optional 3D Gaussian splat gallery with LOD streaming and strict memory/download gate.
 * This component implements the strict download gate and opt-in architecture.
 * Actual splat rendering would use Three.js + a Splat loader (like Luma WebGL or antimatter15's splat),
 * but for this experiment we demonstrate the capability-gated wrapper.
 */
export function GaussianSplatGallery() {
  const [hasOptedIn, setHasOptedIn] = useState(false);
  const [isSupported, setIsSupported] = useState(true);
  const [loadingProgress, setLoadingProgress] = useState(0);
  const [activeLOD, setActiveLOD] = useState<string>('Low');
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Check device capabilities (memory/download gate)
  useEffect(() => {
    // We can use navigator.deviceMemory to gate heavy splats
    const memory = (navigator as any).deviceMemory;
    if (memory && memory < 4) {
      // If the device has less than 4GB of RAM, we might disable it or warn
      setIsSupported(false);
    }
  }, []);

  const handleOptIn = () => {
    setHasOptedIn(true);
    simulateLODStreaming();
  };

  // Simulate a progressive LOD streaming
  const simulateLODStreaming = () => {
    let progress = 0;
    const interval = setInterval(() => {
      progress += 10;
      setLoadingProgress(progress);
      
      if (progress >= 30 && progress < 70) {
        setActiveLOD('Medium');
      } else if (progress >= 70) {
        setActiveLOD('High');
      }

      if (progress >= 100) {
        clearInterval(interval);
        renderSimulatedSplat();
      }
    }, 200);
  };

  const renderSimulatedSplat = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // A simple visual representation of points rendering
    let particles: {x: number, y: number, color: string}[] = [];
    for(let i=0; i<5000; i++) {
      particles.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        color: `hsla(${Math.random() * 60 + 10}, 80%, 50%, ${Math.random()})`
      });
    }

    const draw = () => {
      ctx.fillStyle = 'rgba(15, 23, 42, 0.2)'; // trail effect
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      particles.forEach(p => {
        p.x += (Math.random() - 0.5) * 2;
        p.y += (Math.random() - 0.5) * 2;
        ctx.fillStyle = p.color;
        ctx.fillRect(p.x, p.y, 2, 2);
      });
      requestAnimationFrame(draw);
    };
    draw();
  };

  if (!isSupported) {
    return (
      <div className="border border-slate-700 bg-slate-800 p-6 rounded-lg text-slate-300">
        <h3 className="text-xl font-bold mb-2">IF11: Gaussian Splat Gallery</h3>
        <p className="text-red-400">
          Your device does not meet the strict memory requirements (4GB+ RAM) needed to render volumetric splats.
        </p>
      </div>
    );
  }

  return (
    <div className="border border-slate-700 bg-slate-900 rounded-lg overflow-hidden text-slate-200">
      <div className="p-4 border-b border-slate-800 flex justify-between items-center">
        <h3 className="font-bold">IF11: Gaussian Splat Gallery</h3>
        {hasOptedIn && (
          <div className="text-xs font-mono bg-slate-800 px-2 py-1 rounded">
            LOD: {activeLOD} | Splats: {loadingProgress}%
          </div>
        )}
      </div>

      <div className="relative w-full aspect-video bg-black flex items-center justify-center">
        {!hasOptedIn ? (
          <div className="text-center p-6 bg-slate-800 rounded border border-slate-700 max-w-sm">
            <h4 className="text-lg font-semibold mb-2">Volumetric Data Warning</h4>
            <p className="text-sm text-slate-400 mb-4">
              This gallery contains high-density 3D Gaussian Splats. 
              Downloading this dataset will consume ~45MB of bandwidth and require significant GPU memory.
            </p>
            <button 
              onClick={handleOptIn}
              className="bg-blue-600 hover:bg-blue-500 text-white px-4 py-2 rounded transition-colors"
            >
              Opt-in & Load Splat (45 MB)
            </button>
          </div>
        ) : (
          <canvas 
            ref={canvasRef} 
            width={800} 
            height={450} 
            className="w-full h-full object-cover"
          />
        )}
        
        {hasOptedIn && loadingProgress < 100 && (
          <div className="absolute inset-0 bg-black/80 flex items-center justify-center">
            <div className="w-64">
              <div className="text-sm mb-2 flex justify-between">
                <span>Streaming LOD ({activeLOD})...</span>
                <span>{loadingProgress}%</span>
              </div>
              <div className="w-full bg-slate-800 rounded-full h-2">
                <div 
                  className="bg-blue-500 h-2 rounded-full transition-all duration-200" 
                  style={{ width: `${loadingProgress}%` }}
                />
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
