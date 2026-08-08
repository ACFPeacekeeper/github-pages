import { describe, it, expect } from 'vitest';
import { LinearScale } from '../../../../src/utils/visualization/encodings';

describe('LinearScale', () => {
  it('maps values correctly', () => {
    const scale = new LinearScale([0, 100], [0, 10]);
    expect(scale.map(0)).toBe(0);
    expect(scale.map(50)).toBe(5);
    expect(scale.map(100)).toBe(10);
  });

  it('clamps values to domain', () => {
    const scale = new LinearScale([0, 100], [0, 10]);
    expect(scale.map(-50)).toBe(0);
    expect(scale.map(150)).toBe(10);
  });

  it('handles zero domain range gracefully', () => {
    const scale = new LinearScale([50, 50], [0, 10]);
    expect(scale.map(50)).toBe(0);
  });
});
