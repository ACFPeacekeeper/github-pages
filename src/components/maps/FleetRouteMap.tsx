'use client';

import { useState } from 'react';

const stops = [{ id: 'depot', label: 'Depot', x: 50, y: 50 }, { id: 'north', label: 'North district', x: 28, y: 22 }, { id: 'east', label: 'East district', x: 78, y: 35 }, { id: 'south', label: 'South district', x: 65, y: 78 }];

export default function FleetRouteMap() {
  const [selected, setSelected] = useState('depot');
  const current = stops.find((stop) => stop.id === selected) ?? stops[0];
  return <section className="rounded-2xl border border-emerald-400/20 bg-emerald-950/80 p-5 text-white" aria-labelledby="fleet-map-title"><h2 id="fleet-map-title" className="text-xl font-bold">Fleet coverage map</h2><p className="mt-1 text-sm text-emerald-100">Select a stop to inspect a routing scenario.</p><div className="relative mt-5 aspect-[1.7] overflow-hidden rounded-xl bg-[radial-gradient(circle_at_center,#14532d,#052e16)]"><svg viewBox="0 0 100 100" className="absolute inset-0 h-full w-full" aria-hidden="true"><path d="M50 50 L28 22 M50 50 L78 35 M50 50 L65 78" fill="none" stroke="#6ee7b7" strokeDasharray="3 2" strokeWidth="1.5" /></svg>{stops.map((stop) => <button type="button" key={stop.id} onClick={() => setSelected(stop.id)} className={`absolute -translate-x-1/2 -translate-y-1/2 rounded-full px-2 py-1 text-[.65rem] font-bold ${selected === stop.id ? 'bg-emerald-300 text-emerald-950' : 'bg-emerald-900 text-emerald-100'}`} style={{ left: `${stop.x}%`, top: `${stop.y}%` }} aria-pressed={selected === stop.id}>{stop.label}</button>)}</div><p className="mt-3 text-sm" aria-live="polite">Selected: <strong>{current.label}</strong> · demand window 08:00–12:00</p></section>;
}
