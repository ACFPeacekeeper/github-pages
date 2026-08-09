import { describe, expect, it } from 'vitest';
import { getConnectedNodeIds, getNodeById, isEdgeConnected } from '../../../src/utils/visualization';
import type { ResearchEdge, ResearchNode } from '../../../src/interfaces/visualization';

const nodes: ResearchNode[] = [
  { id: 'a', label: 'A', shortLabel: 'A', description: 'First', domain: 'core', x: 0, y: 0 },
  { id: 'b', label: 'B', shortLabel: 'B', description: 'Second', domain: 'ai', x: 1, y: 1 },
  { id: 'c', label: 'C', shortLabel: 'C', description: 'Third', domain: 'optimization', x: 2, y: 2 },
];
const edges: ResearchEdge[] = [{ source: 'a', target: 'b' }, { source: 'c', target: 'a' }];

describe('visualization graph utilities', () => {
  it('returns the node with the requested id', () => {
    expect(getNodeById(nodes, 'b')?.label).toBe('B');
  });

  it('returns undefined when a node id is absent', () => {
    expect(getNodeById(nodes, 'missing')).toBeUndefined();
  });

  it('collects connections in both edge directions', () => {
    expect(Array.from(getConnectedNodeIds(edges, 'a')).sort()).toEqual(['b', 'c']);
  });

  it('returns an empty set for an isolated node', () => {
    expect(getConnectedNodeIds(edges, 'missing').size).toBe(0);
  });

  it('keeps every edge active when there is no selection', () => {
    expect(isEdgeConnected(edges[0], null)).toBe(true);
  });

  it('dims an edge unrelated to the selection', () => {
    expect(isEdgeConnected(edges[0], 'c')).toBe(false);
  });
});
