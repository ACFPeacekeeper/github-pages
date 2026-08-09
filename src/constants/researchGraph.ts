import type { ResearchGraph } from '../interfaces/visualization';

export const RESEARCH_GRAPH: ResearchGraph = {
  nodes: [
    { id: 'research', label: 'Applied research', shortLabel: 'Research', description: 'The bridge between mathematical ideas and useful systems.', domain: 'core', x: 50, y: 50 },
    { id: 'ai', label: 'Artificial intelligence', shortLabel: 'AI', description: 'Learning representations, policies, and useful heuristics from data.', domain: 'ai', x: 25, y: 35, href: '/github-pages/content/tools' },
    { id: 'or', label: 'Operations research', shortLabel: 'OR', description: 'Models, constraints, exact methods, and principled approximations.', domain: 'optimization', x: 75, y: 35 },
    { id: 'rl_notes', label: 'Notes on RL', shortLabel: 'RL Notes', description: 'Sequential decision-making with learned value functions and policies.', domain: 'ai', x: 15, y: 15, href: '/github-pages/content/posts/Notes_on_RL_an_Introduction' },
    { id: 'or_intro', label: 'Combinatorial Optimization', shortLabel: 'OR Intro', description: 'Foundations and exact methods for hard problems.', domain: 'optimization', x: 85, y: 15, href: '/github-pages/content/posts/Combinatorial_Optimization_an_Introduction' },
    { id: 'routing', label: 'Vehicle routing', shortLabel: 'Routing', description: 'Planning efficient routes under capacity and periodic constraints.', domain: 'optimization', x: 85, y: 55, href: '/github-pages/content/posts/Attention_Learn_to_Solve_Routing_Problem' },
    { id: 'pcvrp', label: 'PCVRP Solver', shortLabel: 'PCVRP', description: 'Periodic Capacitated Vehicle Routing Problem implementation.', domain: 'application', x: 75, y: 80, href: '/github-pages/content/projects/PCVRP' },
    { id: 'audio', label: 'Audio Signal Processing', shortLabel: 'Audio DSP', description: 'Analysis and manipulation of audio signals.', domain: 'application', x: 25, y: 80, href: '/github-pages/content/projects/Audio_Signal_Processing' },
    { id: 'systems', label: 'Intelligent systems', shortLabel: 'Systems', description: 'Deployable tools where learning and optimization work together.', domain: 'application', x: 50, y: 90, href: '/github-pages/content/projects' },
  ],
  edges: [
    { source: 'research', target: 'ai' },
    { source: 'research', target: 'or' },
    { source: 'research', target: 'systems' },
    { source: 'ai', target: 'rl_notes' },
    { source: 'or', target: 'or_intro' },
    { source: 'or', target: 'routing' },
    { source: 'ai', target: 'audio' },
    { source: 'routing', target: 'pcvrp' },
    { source: 'audio', target: 'systems' },
    { source: 'pcvrp', target: 'systems' },
  ],
};
