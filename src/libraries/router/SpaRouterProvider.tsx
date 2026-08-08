'use client';
import React from 'react';
import { BrowserRouter, Route, Routes } from 'react-router-dom';

/**
 * Note: This project uses Next.js App Router for primary routing.
 * This React Router setup is provided for potential isolated SPA components (islands)
 * or specific client-side routing needs that are separate from Next.js routing.
 */
export function SpaRouterProvider({ children }: { children: React.ReactNode }) {
  return (
    <BrowserRouter>
      {children}
    </BrowserRouter>
  );
}
