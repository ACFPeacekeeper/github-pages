/** Homepage card model (shared with src/pages/index.tsx when wired). */
export interface NavCard {
  icon: string;
  title: string;
  description: string;
  to: string;
}

export const HOME_NAV_CARDS: NavCard[] = [
  {
    icon: '📚',
    title: 'Docs',
    description:
      'Architecture, ADRs, the roadmap/changelog, and every feature roadmap under docs/moon/roadmaps/.',
    to: '/docs/ARCHITECTURE',
  },
  {
    icon: '🧭',
    title: 'Guides',
    description:
      'The repository README, contributing guide, and AGENTS.md — curated, not a full repo crawl.',
    to: '/guides',
  },
  {
    icon: '🧩',
    title: 'API Reference',
    description: "TypeDoc-generated reference for lib/'s exported functions and types.",
    to: '/docs/website/api-docs',
  },
  {
    icon: '🎨',
    title: 'Storybook',
    description:
      'Every UI/layout component from src/frameworks/react/components/, rendered live with hand-written prop tables.',
    to: '/storybook/index.html',
  },
];
