import type { App, Plugin } from 'vue';
import { vClickOutside } from './clickOutside';
import { vFocus } from './focus';
import { vIntersect } from './intersect';

/** Registers docs-site custom Vue directives for island mounts. */
export const directivesPlugin: Plugin = {
  install(app: App) {
    app.directive('click-outside', vClickOutside);
    app.directive('focus', vFocus);
    app.directive('intersect', vIntersect);
  },
};

export { vClickOutside, vFocus, vIntersect };
