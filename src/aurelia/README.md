# Aurelia islands

This directory contains optional Aurelia 2 custom elements. They consume framework-neutral logic from `src/simulations/` and must be mounted lazily with `mountAureliaSimulation`; importing an island into the root layout would add Aurelia to every route.

Each island must expose semantic fallback content in the Next.js page, own and stop its Aurelia application lifecycle, avoid global styles/state, and remain compatible with static export. React remains responsible for the shared application shell and routing.
