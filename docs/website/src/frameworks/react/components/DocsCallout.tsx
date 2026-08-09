import React, { type ReactNode } from 'react';

/** Lightweight callout for Docusaurus MDX / local docs UI (host-local, not main-site). */
export function DocsCallout({
  title,
  children,
}: {
  title: string;
  children: ReactNode;
}) {
  return (
    <aside
      style={{
        border: '1px solid var(--ifm-color-emphasis-300)',
        borderRadius: 10,
        padding: '0.85rem 1rem',
        margin: '1rem 0',
        background: 'var(--ifm-color-emphasis-100)',
      }}
    >
      <strong style={{ display: 'block', marginBottom: '0.35rem' }}>{title}</strong>
      <div>{children}</div>
    </aside>
  );
}

export default DocsCallout;
