class AstroIslandElement extends HTMLElement {
  connectedCallback() {
    this.innerHTML = `
      <section aria-label="Astro island">
        <h2>Astro Island</h2>
        <p>This is a simulated Astro island component mounted within Next.js.</p>
      </section>
    `;
  }
}

if (typeof window !== 'undefined' && !customElements.get('astro-island')) {
  customElements.define('astro-island', AstroIslandElement);
}

export function mountAstroIsland(host: HTMLElement): () => void {
  const element = document.createElement('astro-island');
  host.appendChild(element);
  return () => {
    host.removeChild(element);
  };
}
