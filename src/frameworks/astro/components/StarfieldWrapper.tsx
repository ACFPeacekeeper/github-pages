'use client';
import React from 'react';

interface StarfieldWrapperProps {
  height?: string;
}

export function StarfieldWrapper({ height = '300px' }: StarfieldWrapperProps) {
  return (
    <div style={{ padding: '1rem', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}>
      <iframe
        src="/github-pages/astro-island/starfield/index.html"
        style={{
          width: '100%',
          height,
          border: 'none',
          overflow: 'hidden',
          backgroundColor: '#000',
        }}
        title="Astro Starfield"
      />
    </div>
  );
}
