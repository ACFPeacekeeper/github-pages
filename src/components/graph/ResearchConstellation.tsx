'use client';

import { ArrowUpRight, Network } from 'lucide-react';
import { VISUAL_EXPERIENCE } from '../../configs/visualExperience';
import { RESEARCH_GRAPH } from '../../constants/researchGraph';
import type { ResearchDomain } from '../../interfaces/visualization';
import { getConnectedNodeIds, getNodeById, isEdgeConnected } from '../../utils/visualization';
import { useSelection } from '../../utils/visualization/useSelection';
import { useKeyboardRoving } from '../../utils/visualization/useKeyboardRoving';
import { A11ySummary } from '../visualization/A11ySummary';
import { A11yTable } from '../visualization/A11yTable';
import { Legend } from '../visualization/Legend';
import { Shape } from '../visualization/Shape';
import { DEFAULT_DOMAIN_PALETTE } from '../../utils/visualization/encodings';

const domainClasses: Record<ResearchDomain, string> = {
  core: 'constellation-node--core',
  ai: 'constellation-node--ai',
  optimization: 'constellation-node--optimization',
  application: 'constellation-node--application',
};

const LEGEND_ITEMS = [
  { key: 'core', label: 'Core' },
  { key: 'ai', label: 'Artificial Intelligence' },
  { key: 'optimization', label: 'Optimization' },
  { key: 'application', label: 'Application' },
];

export default function ResearchConstellation() {
  const [selectedId, setSelectedId] = useSelection('research', true);
  const containerRef = useKeyboardRoving<HTMLDivElement>('.constellation-node');
  
  const selectedNode = selectedId ? getNodeById(RESEARCH_GRAPH.nodes, selectedId) : undefined;
  const connectedIds = selectedId ? getConnectedNodeIds(RESEARCH_GRAPH.edges, selectedId) : new Set<string>();
  const { width, height } = VISUAL_EXPERIENCE.constellationViewBox;

  return (
    <section className="constellation-panel" aria-labelledby="constellation-title">
      <div className="constellation-copy">
        <div className="eyebrow"><Network aria-hidden="true" size={15} /> Interactive knowledge map</div>
        <h2 id="constellation-title">Research is a connected system.</h2>
        <p>Select a node to trace how learning, optimization, and real-world applications reinforce one another.</p>
        
        <Legend
          title="Research Domains"
          palette={DEFAULT_DOMAIN_PALETTE}
          items={LEGEND_ITEMS}
          selectedKey={selectedNode?.domain}
          className="mt-4 mb-4"
        />

        <div className="constellation-detail" aria-live="polite">
          <span>{selectedNode?.shortLabel ?? 'Explore'}</span>
          <p>{selectedNode?.description ?? 'Choose a research domain to learn more.'}</p>
          {selectedNode?.href && <a href={selectedNode.href}>Explore this thread <ArrowUpRight size={15} aria-hidden="true" /></a>}
        </div>
      </div>

      <A11ySummary
        id="constellation-a11y-summary"
        summary="An interactive node-link diagram showing connections between research domains."
        selectionAnnouncement={selectedNode ? `Selected: ${selectedNode.label}. ${selectedNode.description}` : 'No node selected.'}
      />
      
      <A11yTable
        id="constellation-a11y-table"
        caption="Research nodes and their domains"
        data={RESEARCH_GRAPH.nodes}
        columns={[
          { key: 'label', header: 'Topic' },
          { key: 'domain', header: 'Domain' },
          { key: 'description', header: 'Description' },
        ]}
      />

      <div className="constellation-map" aria-label="Interactive map of research themes" ref={containerRef}>
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
          const encoding = DEFAULT_DOMAIN_PALETTE[node.domain];
          
          return (
            <button
              key={node.id}
              type="button"
              className={`constellation-node ${domainClasses[node.domain]} ${selectedId === node.id ? 'is-selected' : ''} ${selectedId && !isRelated ? 'is-muted' : ''}`}
              style={{ left: `${node.x}%`, top: `${node.y}%`, display: 'flex', alignItems: 'center', gap: '4px' }}
              aria-pressed={selectedId === node.id}
              aria-label={`${node.label}: ${node.description}`}
              onClick={() => setSelectedId(node.id)}
            >
              <Shape encoding={encoding} size={16} />
              {node.shortLabel}
            </button>
          );
        })}
      </div>
    </section>
  );
}
