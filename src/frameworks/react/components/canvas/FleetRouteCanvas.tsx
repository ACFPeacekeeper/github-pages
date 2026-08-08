'use client';

import { useEffect, useRef } from 'react';

export default function FleetRouteCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext('2d');
    if (!context) return;
    const draw = () => {
      const width = canvas.clientWidth * window.devicePixelRatio;
      const height = canvas.clientHeight * window.devicePixelRatio;
      canvas.width = width; canvas.height = height; context.clearRect(0, 0, width, height);
      context.strokeStyle = '#67e8f9'; context.lineWidth = 4 * window.devicePixelRatio; context.lineCap = 'round';
      const routes = [[.1, .75, .3, .25, .62, .58, .88, .18], [.12, .28, .48, .48, .72, .25, .9, .7]];
      routes.forEach((route, index) => { context.beginPath(); context.moveTo(route[0] * width, route[1] * height); for (let i = 2; i < route.length; i += 2) context.lineTo(route[i] * width, route[i + 1] * height); context.strokeStyle = index ? '#a78bfa' : '#67e8f9'; context.stroke(); });
      [[.1, .75], [.3, .25], [.62, .58], [.88, .18], [.12, .28], [.48, .48], [.72, .25], [.9, .7]].forEach(([x, y]) => { context.fillStyle = '#f8fafc'; context.beginPath(); context.arc(x * width, y * height, 5 * window.devicePixelRatio, 0, Math.PI * 2); context.fill(); });
    };
    draw(); window.addEventListener('resize', draw); return () => window.removeEventListener('resize', draw);
  }, []);
  return <figure className="rounded-2xl border border-slate-200 bg-slate-950 p-4 dark:border-white/10"><canvas ref={canvasRef} className="h-56 w-full" aria-label="Two illustrative recycling truck routes connecting a depot and collection points" role="img" /><figcaption className="mt-3 text-xs text-slate-400">A canvas rendering surface for fleet route experiments; the route table remains the authoritative accessible view.</figcaption></figure>;
}
