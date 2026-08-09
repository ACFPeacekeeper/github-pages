import { ExperienceQuality } from '../enums/ExperienceQuality';

/** Tunables for the Docusaurus docs dashboard and Storybook embeds. */
export const DOCS_EXPERIENCE = {
  defaultQuality: ExperienceQuality.Full,
  storybookPath: '/storybook/index.html',
  liveSiteHref: 'https://acfharbinger.github.io/github-pages/',
  researchOrbitHeight: '380px',
} as const;
