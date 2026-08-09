import type { LoreStory } from '../interfaces/types';
import { RESEARCH_OBSERVATORY } from './researchObservatory';
import { MULTI_FRAMEWORK_ISLANDS } from './multiFrameworkIslands';

/** Research/portfolio lore catalog for the docs site. */
export const LORE_STORIES: LoreStory[] = [RESEARCH_OBSERVATORY, MULTI_FRAMEWORK_ISLANDS];

export function getLoreStory(id: string): LoreStory | undefined {
  return LORE_STORIES.find((s) => s.id === id);
}

export { RESEARCH_OBSERVATORY, MULTI_FRAMEWORK_ISLANDS };
