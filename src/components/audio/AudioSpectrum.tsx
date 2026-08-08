'use client';

import { useEffect, useState } from 'react';

const initialBars = [18, 34, 26, 52, 42, 76, 48, 63, 31, 57, 39, 23, 45, 68, 35, 25];

export default function AudioSpectrum() {
  const [playing, setPlaying] = useState(false);
  const [bars, setBars] = useState(initialBars);
  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => setBars(initialBars.map((value) => Math.max(10, Math.min(92, value + Math.round((Math.random() - 0.5) * 24))))), 180);
    return () => window.clearInterval(timer);
  }, [playing]);
  return <section className="rounded-2xl border border-cyan-400/20 bg-slate-950 p-5 text-slate-100" aria-labelledby="audio-spectrum-title">
    <div className="flex items-center justify-between"><div><p className="text-xs font-bold uppercase tracking-[.18em] text-cyan-300">ML signal lab</p><h2 id="audio-spectrum-title" className="mt-2 text-xl font-bold">Audio feature extractor</h2></div><button type="button" onClick={() => setPlaying((value) => !value)} className="rounded-lg border border-cyan-300/30 px-3 py-2 text-xs font-bold">{playing ? 'Pause' : 'Preview'}</button></div>
    <svg className="mt-6 h-28 w-full" viewBox="0 0 160 100" role="img" aria-label="Audio frequency spectrum preview"><g>{bars.map((height, index) => <rect key={index} x={index * 10 + 2} y={100 - height} width="6" height={height} rx="3" fill={index > 9 ? '#a78bfa' : '#22d3ee'} opacity=".8" />)}</g></svg>
    <p className="mt-3 text-xs text-slate-400">A visual companion for signal-processing experiments; production FFT input can replace this deterministic preview.</p>
  </section>;
}
