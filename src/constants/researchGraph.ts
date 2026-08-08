import type { ResearchGraph } from '../interfaces/visualization';

export const RESEARCH_GRAPH: ResearchGraph = {
  nodes: [
    { id: 'research', label: 'Applied research', shortLabel: 'Research', description: 'The bridge between mathematical ideas and useful systems.', domain: 'core', x: 50, y: 48 },
    { id: 'ai', label: 'Artificial intelligence', shortLabel: 'AI', description: 'Learning representations, policies, and useful heuristics from data.', domain: 'ai', x: 23, y: 22, href: '/github-pages/content/tools' },
    { id: 'rl', label: 'Deep reinforcement learning', shortLabel: 'Deep RL', description: 'Sequential decision-making with learned value functions and policies.', domain: 'ai', x: 17, y: 70, href: '/github-pages/content/posts/Notes_on_RL_an_Introduction' },
    { id: 'or', label: 'Operations research', shortLabel: 'OR', description: 'Models, constraints, exact methods, and principled approximations.', domain: 'optimization', x: 77, y: 22, href: '/github-pages/content/posts/Combinatorial_Optimization_an_Introduction' },
    { id: 'routing', label: 'Vehicle routing', shortLabel: 'Routing', description: 'Planning efficient routes under capacity and periodic constraints.', domain: 'optimization', x: 83, y: 70, href: '/github-pages/content/posts/Attention_Learn_to_Solve_Routing_Problem' },
    { id: 'systems', label: 'Intelligent systems', shortLabel: 'Systems', description: 'Deployable tools where learning and optimization work together.', domain: 'application', x: 50, y: 84, href: '/github-pages/content/projects' },
  ],
  edges: [
    { source: 'research', target: 'ai' },
    { source: 'research', target: 'rl' },
    { source: 'research', target: 'or' },
    { source: 'research', target: 'routing' },
    { source: 'research', target: 'systems' },
    { source: 'ai', target: 'rl' },
    { source: 'or', target: 'routing' },
    { source: 'rl', target: 'systems' },
    { source: 'routing', target: 'systems' },
  ],
};
