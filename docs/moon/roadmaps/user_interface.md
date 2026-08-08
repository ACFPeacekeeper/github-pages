# User Interface Roadmap

Planned work on the site's rendering architecture, layout, and motion — see [`docs/moon/ROADMAP.md`](../ROADMAP.md) for the project-level roadmap this rolls up into, and [`docs/moon/research/`](../research/) for the design research behind these items.

| # | Item | Effort | Status |
| --- | --- | --- | --- |
| UI1 | Next.js App Router site with Tailwind CSS styling | M | ✅ Done |
| UI2 | Dark/light theme toggle, persisted across navigation via the sidebar | S | ✅ Done |
| UI3 | Evaluate a persistent canvas layer (single WebGL/Three.js context spanning route changes) instead of remounting per page | M | 📋 Pending |
| UI4 | Adopt the View Transitions API + FLIP-style animations for content and page transitions | M | 📋 Pending |
| UI5 | Revisit hydration strategy (islands/partial hydration) if interactive widgets from `interactive_features.md` land | M | 📋 Pending |

> **TODO:** Reprioritize once `interactive_features.md` items are scoped, since several depend on the rendering architecture decided here first.
