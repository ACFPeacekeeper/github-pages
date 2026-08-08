"use client";

import React, { useEffect, useRef, useState } from "react";

export interface DataPoint {
  id: string;
  x: number;
  y: number;
  label?: string;
}

export interface GeospatialRendererProps {
  data: DataPoint[];
  width?: number;
  height?: number;
  pointRadius?: number;
  color?: string;
  forceSvg?: boolean;
}

const CANVAS_THRESHOLD = 500; // Switch to Canvas if data length exceeds this

/**
 * A hybrid Geospatial/Graph Renderer that uses SVG for small datasets
 * and Canvas 2D for larger datasets to optimize performance.
 */
export const GeospatialRenderer: React.FC<GeospatialRendererProps> = ({
  data,
  width = 600,
  height = 400,
  pointRadius = 3,
  color = "#3b82f6",
  forceSvg = false,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [useCanvas, setUseCanvas] = useState(!forceSvg && data.length > CANVAS_THRESHOLD);

  // Allow manual toggle for demonstration purposes
  const toggleRenderer = () => setUseCanvas((prev) => !prev);

  // Auto-switch based on data size if not forced
  useEffect(() => {
    if (!forceSvg) {
      setUseCanvas(data.length > CANVAS_THRESHOLD);
    } else {
      setUseCanvas(false);
    }
  }, [data.length, forceSvg]);

  // Canvas rendering logic
  useEffect(() => {
    if (!useCanvas || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Draw points
    ctx.fillStyle = color;
    data.forEach((point) => {
      ctx.beginPath();
      ctx.arc(point.x, point.y, pointRadius, 0, 2 * Math.PI);
      ctx.fill();
    });
  }, [data, useCanvas, width, height, pointRadius, color]);

  return (
    <div className="flex flex-col gap-4 p-4 border rounded-lg bg-white shadow-sm dark:bg-gray-800 dark:border-gray-700">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
          Geospatial Graph
        </h3>
        <button
          onClick={toggleRenderer}
          className="px-3 py-1 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          {useCanvas ? "Switch to SVG" : "Switch to Canvas"}
        </button>
      </div>

      <div className="text-sm text-gray-500 dark:text-gray-400">
        Current Renderer: <strong>{useCanvas ? "Canvas 2D" : "SVG"}</strong> (Points: {data.length})
      </div>

      <div 
        className="relative overflow-hidden border border-gray-200 dark:border-gray-600 rounded-md bg-gray-50 dark:bg-gray-900" 
        style={{ width, height }}
      >
        {useCanvas ? (
          <canvas
            ref={canvasRef}
            width={width}
            height={height}
            className="absolute top-0 left-0"
          />
        ) : (
          <svg
            width={width}
            height={height}
            className="absolute top-0 left-0"
          >
            {data.map((point) => (
              <circle
                key={point.id}
                cx={point.x}
                cy={point.y}
                r={pointRadius}
                fill={color}
                className="transition-all duration-300 hover:r-5 hover:fill-blue-800 cursor-pointer"
              >
                {point.label && <title>{point.label}</title>}
              </circle>
            ))}
          </svg>
        )}
      </div>
    </div>
  );
};
