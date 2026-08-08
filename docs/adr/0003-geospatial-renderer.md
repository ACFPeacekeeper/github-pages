# ADR 0003: Geospatial and Graph Renderer Strategy

## Status
Accepted

## Context
We need a strategy for rendering geospatial and graph data visualizations within the site. As the dataset sizes can vary significantly, we must choose a rendering technology that provides the best balance between performance, bundle size, and ease of implementation. WebGL-based solutions like Deck.gl offer extreme performance for large datasets but come with a steep learning curve, large bundle sizes, and complexity. SVG and Canvas 2D are native web technologies that are easier to implement and maintain but have performance limits.

## Decision
We will adopt a progressive enhancement strategy for geospatial and graph rendering:

1. **SVG First:** For small, highly interactive, and easily stylable graphs (typically < 1,000 nodes), we default to SVG. It integrates perfectly with React, CSS, and DOM events.
2. **Canvas 2D as Primary Large-Data Renderer:** For medium to large datasets (1,000 to ~10,000 nodes) where SVG DOM overhead becomes a bottleneck, we will use Canvas 2D. Canvas can efficiently render thousands of points and lines without freezing the browser UI.
3. **WebGL/Deck.gl as Last Resort:** We will only introduce WebGL (e.g., Deck.gl) if measured thresholds prove that Canvas 2D is insufficient for a specific use case (e.g., consistently rendering > 10,000 nodes with high-frequency updates or 3D requirements).

## Consequences
- **Positive:** Keeps the application bundle size small. Development remains simpler using familiar React and Canvas APIs. Reduces the risk of over-engineering early in the project.
- **Negative:** We may need to rewrite a visualization from Canvas to WebGL later if a dataset grows beyond the 10,000 node threshold unexpectedly.
- **Mitigation:** We encapsulate rendering logic behind a generic `GeospatialRenderer` component interface to allow swapping underlying technologies (SVG vs Canvas) without changing the consumer API.
