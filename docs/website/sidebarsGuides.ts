import type { SidebarsConfig } from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  guidesSidebar: [
    {
      type: 'category',
      label: 'Repository Guides',
      collapsed: false,
      items: ['README', 'git/CONTRIBUTING', '.agent/AGENTS'],
    },
  ],
};

export default sidebars;
