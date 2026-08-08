'use client';

import { useState } from 'react';
import { ArrowUpRight, Network } from 'lucide-react';
import { VISUAL_EXPERIENCE } from '../../configs/visualExperience';
import { RESEARCH_GRAPH } from '../../constants/researchGraph';
import type { ResearchDomain } from '../../interfaces/visualization';
import { getConnectedNodeIds, getNodeById, isEdgeConnected } from '../../utils/visualization';

const domainClasses: Record<ResearchDomain, string> = {
  core: 'constellation-node--core',
  ai: 'constellation-node--ai',
  optimization: 'constellation-node--optimization',
  application: 'constellation-node--application',
};

export default function ResearchConstellation() {
  const [selectedId, setSelectedId] = useState<string | null>('research');
  const selectedNode = selectedId ? getNodeById(RESEARCH_GRAPH.nodes, selectedId) : undefined;
  const connectedIds = selectedId ? getConnectedNodeIds(RESEARCH_GRAPH.edges, selectedId) : new Set<string>();
  const { width, height } = VISUAL_EXPERIENCE.constellationViewBox;

  return (
    <section className="constellation-panel" aria-labelledby="constellation-title">
      <div className="constellation-copy">
        <div className="eyebrow"><Network aria-hidden="true" size={15} /> Interactive knowledge map</div>
        <h2 id="constellation-title">Research is a connected system.</h2>
        <p>Select a node to trace how learning, optimization, and real-world applications reinforce one another.</p>
        <div className="constellation-detail" aria-live="polite">
          <span>{selectedNode?.shortLabel ?? 'Explore'}</span>
          <p>{selectedNode?.description ?? 'Choose a research domain to learn more.'}</p>
          {selectedNode?.href && <a href={selectedNode.href}>Explore this thread <ArrowUpRight size={15} aria-hidden="true" /></a>}
        </div>
      </div>

      <div className="constellation-map" aria-label="Interactive map of research themes">
        <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Connections between six research themes">
          <defs>
            <linearGradient id="constellation-line" x1="0" x2="1">
              <stop offset="0" stopColor="#38bdf8" />
              <stop offset="1" stopColor="#a78bfa" />
            </linearGradient>
          </defs>
          {RESEARCH_GRAPH.edges.map((edge) => {
            const source = getNodeById(RESEARCH_GRAPH.nodes, edge.source);
            const target = getNodeById(RESEARCH_GRAPH.nodes, edge.target);
            if (!source || !target) return null;
            return <line key={`${edge.source}-${edge.target}`} x1={source.x * width / 100} y1={source.y * height / 100} x2={target.x * width / 100} y2={target.y * height / 100} className={isEdgeConnected(edge, selectedId) ? 'is-active' : ''} />;
          })}
        </svg>
        {RESEARCH_GRAPH.nodes.map((node) => {
          const isRelated = selectedId === node.id || connectedIds.has(node.id);
          return (
            <button
              key={node.id}
              type="button"
              className={`constellation-node ${domainClasses[node.domain]} ${selectedId === node.id ? 'is-selected' : ''} ${selectedId && !isRelated ? 'is-muted' : ''}`}
              style={{ left: `${node.x}%`, top: `${node.y}%` }}
              aria-pressed={selectedId === node.id}
              aria-label={`${node.label}: ${node.description}`}
              onClick={() => setSelectedId(node.id)}
            >
              <span aria-hidden="true" />
              {node.shortLabel}
            </button>
          );
        })}
      </div>
    </section>
  );
}
