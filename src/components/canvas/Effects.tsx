'use client';

import React, { useEffect, useState, useRef, createContext, useContext } from 'react';

// Preferences Context
export const EffectPreferencesContext = createContext({
  spotlight: true,
  tilt: true,
  particles: true,
  bloomNoise: true,
  distortion: true,
  reducedMotion: false,
  setPreference: (key: string, value: boolean) => {},
});

export const EffectPreferencesProvider = ({ children }: { children: React.ReactNode }) => {
  const [prefs, setPrefs] = useState({
    spotlight: true,
    tilt: true,
    particles: true,
    bloomNoise: true,
    distortion: true,
    reducedMotion: false,
  });

  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    setPrefs(p => ({ ...p, reducedMotion: mediaQuery.matches }));
    
    const listener = (e: MediaQueryListEvent) => {
      setPrefs(p => ({ ...p, reducedMotion: e.matches }));
    };
    mediaQuery.addEventListener('change', listener);
    return () => mediaQuery.removeEventListener('change', listener);
  }, []);

  const setPreference = (key: string, value: boolean) => {
    setPrefs(p => ({ ...p, [key]: value }));
  };

  return (
    <EffectPreferencesContext.Provider value={{ ...prefs, setPreference }}>
      {children}
    </EffectPreferencesContext.Provider>
  );
};

export const useEffectPreferences = () => useContext(EffectPreferencesContext);

export const Effects = () => {
  const prefs = useEffectPreferences();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  // Spotlight
  useEffect(() => {
    if (!prefs.spotlight || prefs.reducedMotion) return;
    
    const handleMouseMove = (e: MouseEvent) => {
      document.documentElement.style.setProperty('--cursor-x', `${e.clientX}px`);
      document.documentElement.style.setProperty('--cursor-y', `${e.clientY}px`);
    };
    
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, [prefs.spotlight, prefs.reducedMotion]);

  // Particles, Noise, Distortion (Canvas)
  useEffect(() => {
    if ((!prefs.particles && !prefs.bloomNoise && !prefs.distortion) || prefs.reducedMotion) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    let animationId: number;
    const particles = Array.from({ length: 50 }).map(() => ({
      x: Math.random() * window.innerWidth,
      y: Math.random() * window.innerHeight,
      vx: (Math.random() - 0.5) * 0.5,
      vy: (Math.random() - 0.5) * 0.5,
      size: Math.random() * 2,
    }));

    const render = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      if (prefs.bloomNoise) {
        ctx.fillStyle = 'rgba(255, 255, 255, 0.02)';
        for (let i = 0; i < 100; i++) {
          ctx.fillRect(Math.random() * canvas.width, Math.random() * canvas.height, 2, 2);
        }
      }

      if (prefs.particles) {
        ctx.fillStyle = 'rgba(200, 200, 255, 0.5)';
        particles.forEach(p => {
          p.x += p.vx;
          p.y += p.vy;
          if (p.x < 0) p.x = canvas.width;
          if (p.x > canvas.width) p.x = 0;
          if (p.y < 0) p.y = canvas.height;
          if (p.y > canvas.height) p.y = 0;
          ctx.beginPath();
          ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
          ctx.fill();
        });
      }

      if (prefs.distortion) {
        // Mock distortion effect by slight scaling/filtering
        ctx.fillStyle = 'rgba(0, 0, 0, 0.01)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
      }

      animationId = requestAnimationFrame(render);
    };
    render();
    return () => cancelAnimationFrame(animationId);
  }, [prefs.particles, prefs.bloomNoise, prefs.distortion, prefs.reducedMotion]);

  return (
    <>
      {prefs.spotlight && !prefs.reducedMotion && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            width: '100vw',
            height: '100vh',
            pointerEvents: 'none',
            zIndex: 9999,
            background: 'radial-gradient(600px circle at var(--cursor-x, 50vw) var(--cursor-y, 50vh), rgba(255,255,255,0.06), transparent 40%)',
          }}
        />
      )}
      {(prefs.particles || prefs.bloomNoise || prefs.distortion) && !prefs.reducedMotion && (
        <canvas
          ref={canvasRef}
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            width: '100vw',
            height: '100vh',
            pointerEvents: 'none',
            zIndex: 9998,
            mixBlendMode: 'screen',
            filter: prefs.distortion ? 'contrast(1.2) brightness(1.1)' : 'none',
          }}
        />
      )}
    </>
  );
};

export const TiltCard = ({ children, className = '' }: { children: React.ReactNode, className?: string }) => {
  const prefs = useEffectPreferences();
  const ref = useRef<HTMLDivElement>(null);

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!prefs.tilt || prefs.reducedMotion || !ref.current) return;
    const rect = ref.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const centerX = rect.width / 2;
    const centerY = rect.height / 2;
    const rotateX = ((y - centerY) / centerY) * -10;
    const rotateY = ((x - centerX) / centerX) * 10;
    ref.current.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) scale3d(1.02, 1.02, 1.02)`;
  };

  const handleMouseLeave = () => {
    if (!ref.current) return;
    ref.current.style.transform = 'perspective(1000px) rotateX(0deg) rotateY(0deg) scale3d(1, 1, 1)';
  };

  return (
    <div
      ref={ref}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      className={className}
      style={{ transition: 'transform 0.1s ease-out', willChange: 'transform' }}
    >
      {children}
    </div>
  );
};

export const EffectsControls = () => {
  const prefs = useEffectPreferences();
  return (
    <div className="p-4 border rounded-lg bg-gray-900 text-white shadow-lg space-y-2 max-w-sm">
      <h3 className="font-bold mb-2">Effect Preferences</h3>
      {Object.entries(prefs).map(([key, val]) => {
        if (key === 'setPreference' || key === 'reducedMotion') return null;
        return (
          <label key={key} className="flex items-center space-x-2">
            <input
              type="checkbox"
              checked={val as boolean}
              onChange={(e) => prefs.setPreference(key, e.target.checked)}
              className="rounded bg-gray-800 border-gray-700 text-blue-500 focus:ring-blue-500"
            />
            <span className="capitalize">{key.replace(/([A-Z])/g, ' $1').trim()}</span>
          </label>
        );
      })}
      {prefs.reducedMotion && (
        <p className="text-sm text-yellow-400 mt-2">Reduced motion is enabled at the OS level. Effects are disabled.</p>
      )}
    </div>
  );
};
