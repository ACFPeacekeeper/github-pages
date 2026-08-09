import React from 'react';
import type { VisualEncoding } from '../../../../utils/visualization/encodings';

interface ShapeProps extends React.SVGProps<SVGSVGElement> {
  encoding: VisualEncoding;
  size?: number;
}

export function Shape({ encoding, size = 16, className, ...props }: ShapeProps) {
  const { color, shape, pattern } = encoding;
  const strokeDasharray = pattern === 'dashed' ? '4 2' : pattern === 'dotted' ? '2 2' : 'none';
  const strokeWidth = 2;
  const half = size / 2;
  const fill = 'currentColor'; // we could use the color directly or via style

  let pathData = '';
  switch (shape) {
    case 'circle':
      return (
        <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className={className} aria-hidden="true" style={{ color }} {...props}>
          <circle cx={half} cy={half} r={half - strokeWidth} fill={fill} stroke={color} strokeWidth={strokeWidth} strokeDasharray={strokeDasharray} />
        </svg>
      );
    case 'square':
      return (
        <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className={className} aria-hidden="true" style={{ color }} {...props}>
          <rect x={strokeWidth} y={strokeWidth} width={size - 2 * strokeWidth} height={size - 2 * strokeWidth} fill={fill} stroke={color} strokeWidth={strokeWidth} strokeDasharray={strokeDasharray} />
        </svg>
      );
    case 'triangle':
      pathData = `M${half},${strokeWidth} L${size - strokeWidth},${size - strokeWidth} L${strokeWidth},${size - strokeWidth} Z`;
      break;
    case 'diamond':
      pathData = `M${half},${strokeWidth} L${size - strokeWidth},${half} L${half},${size - strokeWidth} L${strokeWidth},${half} Z`;
      break;
    case 'hexagon':
      const q = size / 4;
      pathData = `M${half},${strokeWidth} L${size - strokeWidth},${q} L${size - strokeWidth},${size - q} L${half},${size - strokeWidth} L${strokeWidth},${size - q} L${strokeWidth},${q} Z`;
      break;
    default:
      // default to circle
      return (
        <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className={className} aria-hidden="true" style={{ color }} {...props}>
          <circle cx={half} cy={half} r={half - strokeWidth} fill={fill} stroke={color} strokeWidth={strokeWidth} strokeDasharray={strokeDasharray} />
        </svg>
      );
  }

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className={className} aria-hidden="true" style={{ color }} {...props}>
      <path d={pathData} fill={fill} stroke={color} strokeWidth={strokeWidth} strokeDasharray={strokeDasharray} />
    </svg>
  );
}
