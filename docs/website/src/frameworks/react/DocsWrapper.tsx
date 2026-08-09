'use client';

import React, { useEffect, useId, useRef } from 'react';
import { generateIslandId, logIslandMount, logIslandUnmount } from '../shared/utils';

export interface DocsWrapperProps {
  /** Raw HTML content string to render inside the island card. */
  content: string;
  /** Optional heading shown above the content body. */
  title?: string;
  /** Optional className for the outer card. */
  className?: string;
}

/**
 * Load MathJax once and typeset the page (same approach as main-site PostWrapper).
 */
function loadMathJax(): void {
  if (typeof window === 'undefined') return;

  if (typeof window.MathJax === 'undefined') {
    window.MathJax = {
      tex: {
        inlineMath: [
          ['$', '$'],
          ['\\(', '\\)'],
        ],
        displayMath: [
          ['$$', '$$'],
          ['\\[', '\\]'],
        ],
      },
      svg: {
        fontCache: 'global',
      },
    };
  }

  if (typeof window.MathJax?.typeset === 'function') {
    window.MathJax.typesetClear?.();
    window.MathJax.typeset();
    return;
  }

  if (document.getElementById('MathJax-script')) return;

  const mathJaxScript = document.createElement('script');
  mathJaxScript.id = 'MathJax-script';
  mathJaxScript.async = true;
  mathJaxScript.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js';
  document.head.appendChild(mathJaxScript);
}

/**
 * Docs-site React content wrapper — analogue of main-site `PostWrapper` /
 * `ReportWrapper` under `src/frameworks/react/`.
 *
 * Renders HTML content in a card, typesets math with MathJax, and logs
 * island mount/unmount via shared framework helpers.
 */
const DocsWrapper: React.FC<DocsWrapperProps> = ({ content, title, className }) => {
  const reactId = useId();
  const islandIdRef = useRef(generateIslandId('react-docs'));
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const islandId = islandIdRef.current;
    logIslandMount('React', islandId);

    if (content) {
      const timer = window.setTimeout(() => {
        loadMathJax();
      }, 50);
      return () => {
        window.clearTimeout(timer);
        logIslandUnmount('React', islandId);
      };
    }

    return () => {
      logIslandUnmount('React', islandId);
    };
  }, [content]);

  return (
    <div
      id={islandIdRef.current}
      data-react-id={reactId}
      className={className}
      style={{
        background: 'var(--ifm-background-surface-color, var(--ifm-background-color))',
        borderRadius: 12,
        boxShadow: '0 8px 24px rgba(15, 23, 42, 0.08)',
        overflow: 'hidden',
        border: '1px solid var(--ifm-color-emphasis-300)',
      }}
    >
      {title ? (
        <header
          style={{
            padding: '1rem 1.25rem 0',
          }}
        >
          <p
            style={{
              margin: '0 0 0.25rem',
              fontSize: '0.72rem',
              letterSpacing: '0.08em',
              textTransform: 'uppercase',
              opacity: 0.7,
            }}
          >
            Framework island · React
          </p>
          <h2 style={{ margin: 0, fontSize: '1.25rem' }}>{title}</h2>
        </header>
      ) : null}
      <div
        ref={containerRef}
        className="docs-wrapper-container"
        style={{ padding: '1.25rem 1.5rem 1.5rem' }}
        dangerouslySetInnerHTML={{ __html: content }}
      />
    </div>
  );
};

export default DocsWrapper;

declare global {
  interface Window {
    MathJax:
      | {
          tex: {
            inlineMath: string[][];
            displayMath: string[][];
          };
          svg: {
            fontCache: string;
          };
          typesetClear?: () => void;
          typeset?: () => void;
        }
      | undefined;
  }
}
