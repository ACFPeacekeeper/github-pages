'use client';

import { useEffect, useRef } from 'react';

export function useKeyboardRoving<T extends HTMLElement = HTMLDivElement>(
  selector: string,
  orientation: 'horizontal' | 'vertical' | 'both' = 'both'
) {
  const containerRef = useRef<T>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      const focusableElements = Array.from(
        container.querySelectorAll<HTMLElement>(selector)
      ).filter(el => !el.hasAttribute('disabled'));

      if (focusableElements.length === 0) return;

      const currentIndex = focusableElements.findIndex(el => el === document.activeElement);
      if (currentIndex === -1) return;

      let nextIndex = currentIndex;
      let handled = false;

      switch (e.key) {
        case 'ArrowRight':
          if (orientation === 'horizontal' || orientation === 'both') {
            nextIndex = (currentIndex + 1) % focusableElements.length;
            handled = true;
          }
          break;
        case 'ArrowLeft':
          if (orientation === 'horizontal' || orientation === 'both') {
            nextIndex = (currentIndex - 1 + focusableElements.length) % focusableElements.length;
            handled = true;
          }
          break;
        case 'ArrowDown':
          if (orientation === 'vertical' || orientation === 'both') {
            nextIndex = (currentIndex + 1) % focusableElements.length;
            handled = true;
          }
          break;
        case 'ArrowUp':
          if (orientation === 'vertical' || orientation === 'both') {
            nextIndex = (currentIndex - 1 + focusableElements.length) % focusableElements.length;
            handled = true;
          }
          break;
        case 'Home':
          nextIndex = 0;
          handled = true;
          break;
        case 'End':
          nextIndex = focusableElements.length - 1;
          handled = true;
          break;
      }

      if (handled) {
        e.preventDefault();
        focusableElements[nextIndex].focus();
      }
    };

    container.addEventListener('keydown', handleKeyDown);
    return () => container.removeEventListener('keydown', handleKeyDown);
  }, [selector, orientation]);

  return containerRef;
}
