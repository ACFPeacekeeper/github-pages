import React from 'react';

interface A11ySummaryProps {
  id: string;
  summary: string;
  selectionAnnouncement?: string;
  className?: string;
}

export function A11ySummary({ id, summary, selectionAnnouncement, className = 'sr-only' }: A11ySummaryProps) {
  return (
    <div id={id} className={className}>
      <p>{summary}</p>
      <p aria-live="polite">{selectionAnnouncement}</p>
    </div>
  );
}
