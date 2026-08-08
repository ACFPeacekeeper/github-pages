import React from 'react';
import type { CategoricalPalette } from '../../../../utils/visualization/encodings';
import { Shape } from './Shape';

interface LegendProps<T extends string> {
  title: string;
  palette: CategoricalPalette<T>;
  items: Array<{ key: T; label: string }>;
  onHover?: (key: T | null) => void;
  onSelect?: (key: T) => void;
  selectedKey?: T | null;
  className?: string;
}

export function Legend<T extends string>({
  title,
  palette,
  items,
  onHover,
  onSelect,
  selectedKey,
  className = '',
}: LegendProps<T>) {
  return (
    <fieldset className={`vis-legend ${className}`}>
      <legend className="sr-only">{title}</legend>
      <div className="vis-legend-items" aria-hidden="true">
        <span className="vis-legend-title">{title}</span>
        {items.map((item) => {
          const encoding = palette[item.key];
          const isSelected = selectedKey === item.key;
          const isMuted = selectedKey && !isSelected;
          
          return (
            <button
              key={item.key}
              type="button"
              className={`vis-legend-item ${isSelected ? 'is-selected' : ''} ${isMuted ? 'is-muted' : ''}`}
              onClick={() => onSelect?.(item.key)}
              onMouseEnter={() => onHover?.(item.key)}
              onMouseLeave={() => onHover?.(null)}
              aria-pressed={isSelected}
            >
              <Shape encoding={encoding} size={14} className="vis-legend-shape" />
              <span className="vis-legend-label">{item.label}</span>
            </button>
          );
        })}
      </div>
    </fieldset>
  );
}
