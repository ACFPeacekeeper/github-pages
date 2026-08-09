import { createApp, type App, type Component } from 'vue';
import { directivesPlugin } from './directives';
import { logIslandMount, logIslandUnmount } from '../shared/utils';

/**
 * Mount a Vue island into a host element (docs-site multi-framework slot).
 * Returns an unmount function for React/Docusaurus host lifecycle cleanup.
 */
export function mountVueIsland(
  host: HTMLElement,
  component: Component,
  props: Record<string, unknown> = {},
  islandId = 'vue-island'
): () => void {
  const app: App = createApp(component, props);
  app.use(directivesPlugin);
  app.mount(host);
  logIslandMount('Vue', islandId);

  return () => {
    app.unmount();
    logIslandUnmount('Vue', islandId);
  };
}
