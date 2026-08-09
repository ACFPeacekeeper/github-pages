# Interactive Component Catalogue

This document serves as the registry for all interactive components across the repository. Each domain-specific component is documented below according to the IF13 standard: its input contract, data source, accessibility equivalent, loading behavior, and resource ownership.

## 1. Audio Domain

### `AudioExhibit` (`src/frameworks/react/components/audio/AudioExhibit.tsx`)
*   **Input Contract**: Takes no props (currently standalone demo).
*   **Data Source**: Local media file or synthetic oscillator (demo mode).
*   **Accessibility Equivalent**: Textual state readout ("Playing", "Paused") and visually-hidden status alerts. 
*   **Loading Behavior**: Lazy-loaded `AudioContext` only initialized upon explicit user gesture (Play button click).
*   **Resource Ownership**: Owns a Web Audio `AudioContext`, an `AnalyserNode`, and an HTML `<canvas>` for FFT rendering.

### `AudioSpectrum` (`src/frameworks/react/components/audio/AudioSpectrum.tsx`)
*   **Input Contract**: None (demo display).
*   **Data Source**: Procedurally generated visual frequency data.
*   **Accessibility Equivalent**: Described as a decorative ambient element via `aria-hidden` or `aria-label`.
*   **Loading Behavior**: Rendered immediately on the client.
*   **Resource Ownership**: Does not own hardware resources (pure CSS/DOM animation).

---

## 2. Books Domain

### `ResearchShelf` (`src/frameworks/react/components/books/ResearchShelf.tsx`)
*   **Input Contract**: Optional array of book objects (title, author, cover image).
*   **Data Source**: Static JSON/Mock data.
*   **Accessibility Equivalent**: Semantic `<ul>` list of books with standard `<img alt="...">` tags.
*   **Loading Behavior**: Immediately rendered, images lazily decoded by browser.
*   **Resource Ownership**: No heavy resources.

---

## 3. Canvas Domain

### `Effects` (`src/frameworks/react/components/canvas/Effects.tsx`)
*   **Input Contract**: Preference flags for `reducedMotion`, `bloom`, `noise`, etc.
*   **Data Source**: None (algorithmic).
*   **Accessibility Equivalent**: Completely bypassable via `prefers-reduced-motion` media queries.
*   **Loading Behavior**: Dynamically checks user preferences before rendering the Canvas.
*   **Resource Ownership**: Owns an HTML `<canvas>` and a `requestAnimationFrame` loop.

### `FleetRouteCanvas` (`src/frameworks/react/components/canvas/FleetRouteCanvas.tsx`)
*   **Input Contract**: `routeData` array of vehicle paths.
*   **Data Source**: Computed optimization outputs from the backend/mock data.
*   **Accessibility Equivalent**: Semantic data table of route coordinates and vehicle assignments.
*   **Loading Behavior**: Loaded via Intersection Observer when scrolled into view.
*   **Resource Ownership**: Owns a 2D `<canvas>` context for high-performance rendering.

---

## 4. Graph Domain

### `ResearchConstellation` (`src/frameworks/react/components/graph/ResearchConstellation.tsx`)
*   **Input Contract**: `nodes` and `links` arrays.
*   **Data Source**: `src/constants/researchGraph.ts`.
*   **Accessibility Equivalent**: Keyboard-roving list of nodes with ARIA live announcements for selection. 
*   **Loading Behavior**: Immediately rendered SVG elements.
*   **Resource Ownership**: Owns SVG nodes, no heavy WebGL/Canvas context.

---

## 5. Maps Domain

### `GeospatialRenderer` (`src/frameworks/react/components/maps/GeospatialRenderer.tsx`)
*   **Input Contract**: `nodes` (features), `links` (edges), and a threshold limit.
*   **Data Source**: Geographic datasets (e.g., PCVRP instances).
*   **Accessibility Equivalent**: Data table summary of locations and distances.
*   **Loading Behavior**: Progressively enhanced based on node count (see ADR 0003).
*   **Resource Ownership**: Owns a `<canvas>` for datasets > 500 points, or purely SVG for smaller datasets.

---

## 6. Models Domain

### `HeroModel` (`src/frameworks/react/components/models/HeroModel.tsx`)
*   **Input Contract**: None (hardcoded geometry).
*   **Data Source**: Procedurally generated Three.js primitives.
*   **Accessibility Equivalent**: Fallback static image and descriptive alt text.
*   **Loading Behavior**: Isolated lazy island—Three.js imported only when visible.
*   **Resource Ownership**: Owns a WebGL context (`WebGLRenderer`) and active requestAnimationFrame loop.

### `ModelViewer` (`src/frameworks/react/components/models/ModelViewer.tsx`)
*   **Input Contract**: `modelUrl` (glTF/GLB path).
*   **Data Source**: External `.glb` files.
*   **Accessibility Equivalent**: Annotations are rendered as standard DOM elements overlaying the canvas.
*   **Loading Behavior**: Lazy-loads the `GLTFLoader` and model file.
*   **Resource Ownership**: Owns a WebGL context, geometry buffers, and textures.

### `PanoramaViewer` (`src/frameworks/react/components/models/PanoramaViewer.tsx`)
*   **Input Contract**: `textureUrl` for equirectangular image.
*   **Data Source**: External `.jpg` or `.png` panorama textures.
*   **Accessibility Equivalent**: Flat image fallback with standard scroll navigation.
*   **Loading Behavior**: IntersectionObserver triggered WebGL initialization.
*   **Resource Ownership**: Owns a WebGL context and large texture buffers.

---

## 7. Routes / Simulations Domain

### `ConvergenceSimulation` (`src/frameworks/react/components/routes/ConvergenceSimulation.tsx`)
*   **Input Contract**: Simulation ID string.
*   **Data Source**: Simulation engine (`src/simulations/`).
*   **Accessibility Equivalent**: ARIA live regions announcing current iteration, incumbent, and lower bound.
*   **Loading Behavior**: React state-driven, initialized synchronously.
*   **Resource Ownership**: No hardware resources, pure React/SVG.

---

## 8. Video / Media Domain

### `MediaReel` (`src/frameworks/react/components/video/MediaReel.tsx`)
*   **Input Contract**: Array of video source URLs.
*   **Data Source**: Local `/public` videos or remote URLs.
*   **Accessibility Equivalent**: Native `<video>` controls, captions (VTT) if available, and transcript links.
*   **Loading Behavior**: Uses native `preload="metadata"`.
*   **Resource Ownership**: Owns HTML `<video>` elements.

### `MediaMosaic` (`src/frameworks/react/components/image/MediaMosaic.tsx`)
*   **Input Contract**: Array of image URLs.
*   **Data Source**: Local `/public` images.
*   **Accessibility Equivalent**: `alt` tags and a linear reading order for screen readers.
*   **Loading Behavior**: Native `loading="lazy"` on images.
*   **Resource Ownership**: None.

---

## 9. Games Domain

### `PrototypeCard` (`src/frameworks/react/components/games/PrototypeCard.tsx`)
*   **Input Contract**: Prototype metadata (title, description).
*   **Data Source**: Static JSON.
*   **Accessibility Equivalent**: Standard text descriptions.
*   **Loading Behavior**: Synchronous React render.
*   **Resource Ownership**: None.
