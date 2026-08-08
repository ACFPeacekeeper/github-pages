'use client';

import { useState, useEffect, useCallback } from 'react';

export function useSelection(initialSelection: string | null = null, syncWithHash = false) {
  const [selectedId, setSelectedId] = useState<string | null>(() => {
    if (syncWithHash && typeof window !== 'undefined') {
      const hash = window.location.hash.replace('#', '');
      if (hash) return hash;
    }
    return initialSelection;
  });

  const setSelection = useCallback((id: string | null) => {
    setSelectedId(id);
    if (syncWithHash && typeof window !== 'undefined') {
      const url = new URL(window.location.href);
      if (id) {
        url.hash = id;
      } else {
        url.hash = '';
      }
      // Use replaceState to avoid overwriting browser navigation semantics with too many history entries
      window.history.replaceState(null, '', url.toString());
    }
  }, [syncWithHash]);

  useEffect(() => {
    if (!syncWithHash) return;

    const handleHashChange = () => {
      const hash = window.location.hash.replace('#', '');
      setSelectedId(hash || null);
    };

    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, [syncWithHash]);

  return [selectedId, setSelection] as const;
}
