import type { Directive } from 'vue';

/** Focuses the host when mounted (or when the binding becomes truthy). */
export const vFocus: Directive<HTMLElement, boolean | undefined> = {
  mounted(el, binding) {
    if (binding.value === false) return;
    queueMicrotask(() => el.focus());
  },
  updated(el, binding) {
    if (binding.value && !binding.oldValue) {
      queueMicrotask(() => el.focus());
    }
  },
};
