'use client';

import React, { useEffect, useRef, useState } from 'react';
import { generateIslandId, logIslandMount } from '../../shared/utils';

/**
 * React/Docusaurus host wrapper that dynamically mounts the Aurelia island.
 */
export function AureliaWrapper() {
  const containerRef = useRef<HTMLDivElement>(null);
  const [islandId] = useState(() => generateIslandId('aurelia'));

  useEffect(() => {
    let unmount: (() => Promise<void>) | undefined;
    let isMounted = true;

    async function loadAndMount() {
      if (!containerRef.current) return;
      const { mountAureliaSimulation } = await import('../mount');
      if (!isMounted) return;
      unmount = await mountAureliaSimulation(containerRef.current, islandId);
      logIslandMount('Aurelia', islandId);
    }

    loadAndMount();

    return () => {
      isMounted = false;
      void unmount?.();
    };
  }, [islandId]);

  return (
    <div
      id={islandId}
      ref={containerRef}
      style={{
        padding: '1rem',
        border: '1px solid rgba(148, 163, 184, 0.25)',
        borderRadius: 8,
        minHeight: 150,
      }}
    />
  );
}

export default AureliaWrapper;
