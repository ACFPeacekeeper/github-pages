import React from 'react';
import useBaseUrl from '@docusaurus/useBaseUrl';

/**
 * Iframe host for the prebuilt Astro ResearchOrbit island under static/astro-island.
 */
export function ResearchOrbitWrapper({
  height = '380px',
  title = 'Astro research orbit island',
}: {
  height?: string;
  title?: string;
}) {
  const src = useBaseUrl('/astro-island/index.html');

  return (
    <section style={{ margin: '1.5rem 0 2rem' }}>
      <p
        style={{
          margin: '0 0 0.35rem',
          fontSize: '0.72rem',
          letterSpacing: '0.08em',
          textTransform: 'uppercase',
          opacity: 0.7,
        }}
      >
        Framework island · Astro
      </p>
      <h2 style={{ margin: '0 0 0.5rem', fontSize: '1.25rem' }}>Research orbit</h2>
      <p style={{ margin: '0 0 0.75rem', opacity: 0.8, maxWidth: '40rem' }}>
        Static Astro island illustrating research themes in orbit — multi-framework companion for the docs
        dashboard (MFP5).
      </p>
      <iframe
        src={src}
        title={title}
        style={{
          width: '100%',
          height,
          border: '1px solid var(--ifm-color-emphasis-300)',
          borderRadius: 14,
          background: 'transparent',
        }}
        loading="lazy"
      />
    </section>
  );
}

export default ResearchOrbitWrapper;
