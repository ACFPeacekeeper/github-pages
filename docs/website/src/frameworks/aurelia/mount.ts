import { Aurelia } from 'aurelia';
import { DocsConvergenceApp } from './docs-convergence-app';
import { logIslandMount, logIslandUnmount } from '../shared/utils';

/**
 * Mount the Aurelia docs-site simulation island into a host element.
 * Returns an async unmount for React host lifecycle cleanup.
 */
export async function mountAureliaSimulation(
  host: HTMLElement,
  islandId = 'aurelia-island'
): Promise<() => Promise<void>> {
  const application = new Aurelia();
  application.app({ host, component: DocsConvergenceApp });
  await application.start();
  logIslandMount('Aurelia', islandId);

  return async () => {
    await application.stop(true);
    logIslandUnmount('Aurelia', islandId);
  };
}
