export type ResearchDomain = 'core' | 'ai' | 'optimization' | 'application';

export interface ResearchNode {
  id: string;
  label: string;
  shortLabel: string;
  description: string;
  domain: ResearchDomain;
  x: number;
  y: number;
  href?: string;
}

export interface ResearchEdge {
  source: ResearchNode['id'];
  target: ResearchNode['id'];
}

export interface ResearchGraph {
  nodes: ResearchNode[];
  edges: ResearchEdge[];
}
