import type { ResearchEdge, ResearchNode } from '../interfaces/visualization';

export function getNodeById(nodes: ResearchNode[], id: string): ResearchNode | undefined {
  return nodes.find((node) => node.id === id);
}

export function getConnectedNodeIds(edges: ResearchEdge[], id: string): Set<string> {
  return new Set(
    edges.flatMap((edge) => {
      if (edge.source === id) return [edge.target];
      if (edge.target === id) return [edge.source];
      return [];
    }),
  );
}

export function isEdgeConnected(edge: ResearchEdge, selectedId: string | null): boolean {
  return selectedId === null || edge.source === selectedId || edge.target === selectedId;
}
