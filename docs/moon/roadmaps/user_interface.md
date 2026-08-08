# Visual Design and User Interface Roadmap

Goal: create a distinctive “research observatory” visual identity—deep-space surfaces, luminous data accents, editorial typography, tactile glass panels, and restrained cinematic motion—while keeping content legible and navigation predictable.

| ID | Deliverable | Effort | Depends on | Status |
| --- | --- | --- | --- | --- |
| UI1 | Next.js App Router, responsive shell, content routes and Tailwind foundation | M | — | ✅ |
| UI2 | Persisted light/dark theme and collapsible responsive navigation | S | UI1 | ✅ |
| UI3 | Semantic design tokens for color, type scale, spacing, radius, elevation, blur, motion and chart palettes | M | UI1 | 🚧 |
| UI4 | Layered ambient backdrop: aurora gradients, subtle grid/noise, spotlight response and high-contrast fallback | M | UI3 | 🚧 |
| UI5 | Responsive home composition with editorial hero, proof points, bento cards and clear narrative hierarchy | M | UI3 | 🚧 |
| UI6 | Capability and preference policy covering reduced motion, reduced data, contrast, pointer type, WebGL/WebGPU and device tier | M | UI3, IT6 | 📋 |
| UI7 | Motion language for hover/focus, reveal, shared-element/FLIP project filtering and View Transitions enhancement | L | UI6 | 📋 |
| UI8 | Command palette and global search across projects, posts, reports and tools with keyboard-first navigation | L | DOC5 | 📋 |
| UI9 | Project explorer with URL-backed filters, animated bento layout, comparison mode and accessible list/table view | L | UI7, DOC5 | 📋 |
| UI10 | Case-study template with scrollytelling chapters, sticky metrics, architecture diagrams and media gallery | L | UI7, DOC5 | 📋 |
| UI11 | Accessibility hardening: skip links, focus restoration, landmark audit, contrast modes and screen-reader announcements | M | UI3–UI10 | 📋 |
| UI12 | Internationalization-ready content/layout primitives and locale-safe dates/numbers without requiring an immediate translation | M | DOC5 | 🔬 |
| UI13 | Domain component taxonomy under `src/components/{audio,books,canvas,games,graph,image,maps,models,routes,video}` with shared UI primitives and no catch-all interactive folder | M | UI3, DOC5 | 🚧 |
| UI14 | Redux store for genuinely cross-cutting experience state: theme, quality tier, active simulation and active media; local state remains local | M | UI6, UI13 | 🚧 |

## Acceptance criteria

### UI3–UI5 — visual foundation

- Tokens are CSS custom properties consumed by Tailwind-compatible classes; components do not invent arbitrary repeated colors.
- Light and dark palettes meet WCAG AA contrast for text and controls. Decorative glow never carries meaning.
- Hero scales from 320 px to ultrawide displays without clipped content, horizontal scrolling, or a layout shift.
- Ambient effects use transform/opacity where possible, are `pointer-events: none`, and become static under `prefers-reduced-motion`.
- Visual regression reference images cover home at mobile/tablet/desktop in both themes.

### UI6–UI7 — capabilities and motion

- A typed capability object selects `static`, `reduced`, or `full` presentation and can be overridden by the visitor.
- Native scroll remains the baseline. Any smoothing is optional, does not trap focus, and is disabled for reduced motion.
- Route and layout transitions preserve focus, scroll expectations, browser history, and deep links.
- Animation durations, easing curves, and stagger limits are tokens; continuous animation pauses offscreen or when the document is hidden.

### UI8–UI10 — discovery and storytelling

- Search opens with a documented shortcut, traps focus correctly, returns grouped results, and works entirely from build-generated static data.
- Filter state serializes to query parameters; every visual result is represented in an accessible DOM list.
- Case studies support problem, constraints, approach, architecture, experiment, outcome, limitations, and related-work blocks.
- Interactive embeds reserve aspect ratio, lazy load, expose a text summary, and offer restart/pause controls.

### UI13–UI14 — component and state boundaries

- Each domain directory owns a focused component and tests; components import shared contracts from `src/interfaces` and shared visual primitives from `src/components/ui`.
- `src/redux` is limited to state that crosses routes or independent interactive surfaces. Simulation cursors, hover state, and form drafts stay in the owning component/controller.
- Redux actions are discriminated and serializable; the provider is mounted once at the client shell, and persistence is browser-guarded for static export.

## Design QA matrix

Validate Chromium, Firefox, and WebKit at 320/768/1440 px; keyboard-only; VoiceOver/NVDA smoke paths; 200% zoom; forced colors; reduced motion; coarse pointer; slow 4G; and WebGL unavailable/context-lost states.
