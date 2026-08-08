'use client';
import React, { useEffect, useRef, useState } from 'react';
import { logIslandMount, generateIslandId } from '../shared/utils';

export function AureliaWrapper() {
  const containerRef = useRef<HTMLDivElement>(null);
  const [islandId] = useState(() => generateIslandId('aurelia'));

  useEffect(() => {
    let unmount: (() => Promise<void>) | undefined;
    let isMounted = true;

    async function loadAndMount() {
      if (containerRef.current) {
        const { mountAureliaSimulation } = await import('./mount');
        if (isMounted) {
          unmount = await mountAureliaSimulation(containerRef.current);
          logIslandMount('Aurelia', islandId);
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
  }, [islandId]);

  return <div id={islandId} ref={containerRef} style={{ padding: '1rem', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', minHeight: '150px' }} />;
}
