# 2. Graphics Renderer Lifecycle

Date: 2026-08-08

## Status

Accepted

## Context

The repository roadmap demands rich interactive graphics, including 3D model viewers and geospatial visualizations, while operating under strict performance and bundle size constraints (e.g., initial route JS ≤ 200 kB gzip, LCP ≤ 2.5s). We needed to decide on a lifecycle strategy for WebGL/Three.js renderers.

The primary options were:
1. **Persistent Singleton Canvas:** A single global `<canvas>` element mounted high in the Next.js layout tree, acting as a portal for all graphics across different routes.
2. **Isolated Lazy Islands:** Individual components (e.g., `HeroModel`) that instantiate their own `<canvas>` and WebGL context on demand, dynamically importing the required libraries only when they enter the viewport.

### Evaluation Criteria
- **Memory & Context Count:** Browsers strictly limit the number of concurrent WebGL contexts (usually 8-16). A singleton canvas guarantees only one context is ever created. Isolated islands risk hitting the context limit if multiple canvases exist simultaneously without proper lifecycle management.
- **Route Persistence:** A singleton canvas can maintain state and smooth transitions across Next.js route changes. Islands must be torn down and rebuilt when routes change.
- **First Interaction Cost & Chunk Size:** A singleton canvas requires loading the Three.js library in the global bundle, negatively impacting initial load times for all users, even those who might not scroll to the interactive element. Islands can be code-split and loaded lazily.

## Decision

We will use **Isolated Lazy Islands** with strict lifecycle management.

We will not use a persistent singleton canvas because it violates our strict bundle budget and capabilities-gating policy ("explain before embellishing"). Loading graphics engines globally hurts the initial load time for visitors who may only want to read text.

To mitigate the context limit and memory issues inherent to isolated islands, every graphics component *must* adhere to the lifecycle implemented in `HeroModel` (IF2):
- **Lazy Initialization:** Engines like Three.js are dynamically imported via `IntersectionObserver` only when the component is visible.
- **Visibility Suspension:** Rendering loops must halt when `document.visibilityState === 'hidden'` or the component scrolls out of view.
- **Rigorous Teardown:** The component must explicitly dispose of `WebGLRenderer`, `geometries`, and `materials`, and remove event listeners on unmount.
- **Context Loss Handling:** All components must handle `webglcontextlost` gracefully and provide a static fallback.

## Consequences

- **Positive:** Initial route bundles remain small. Users on constrained devices or low-bandwidth connections are not penalized by heavy graphics engines they haven't requested.
- **Positive:** Components remain modular, isolated, and domain-specific (IF13).
- **Negative:** We lose the ability to animate seamlessly across route transitions.
- **Negative:** Developer overhead increases. Every new graphics component must duplicate boilerplate for visibility checking and WebGL resource disposal to avoid memory leaks and context exhaustion.
