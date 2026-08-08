'use client';

import React, { useEffect, useRef } from 'react';

export function AstroWrapper() {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let unmount: (() => void) | undefined;
    let isMounted = true;

    async function loadAndMount() {
      if (containerRef.current) {
        const { mountAstroIsland } = await import('../astro/mount');
        if (isMounted) {
          unmount = mountAstroIsland(containerRef.current);
        }
      }
    }

    loadAndMount();

    return () => {
      isMounted = false;
      if (unmount) {
        unmount();
      }
    };
  }, []);

  return <div ref={containerRef} />;
}
