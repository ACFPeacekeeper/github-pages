export interface VisualEncoding {
  color: string;
  shape: 'circle' | 'square' | 'triangle' | 'diamond' | 'hexagon';
  pattern?: 'solid' | 'dashed' | 'dotted';
}

export type CategoricalPalette<T extends string | number | symbol> = Record<T, VisualEncoding>;

export const DEFAULT_DOMAIN_PALETTE: CategoricalPalette<string> = {
  core: { color: '#94a3b8', shape: 'circle', pattern: 'solid' },
  ai: { color: '#38bdf8', shape: 'square', pattern: 'solid' },
  optimization: { color: '#a78bfa', shape: 'triangle', pattern: 'solid' },
  application: { color: '#34d399', shape: 'diamond', pattern: 'solid' },
};

export class LinearScale {
  constructor(public domain: [number, number], public range: [number, number]) {}
  
  map(value: number): number {
    const [dMin, dMax] = this.domain;
    const [rMin, rMax] = this.range;
    if (dMax === dMin) return rMin;
    const clampedValue = Math.max(dMin, Math.min(dMax, value));
    const t = (clampedValue - dMin) / (dMax - dMin);
    return rMin + t * (rMax - rMin);
  }
}
