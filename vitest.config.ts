import path from 'path';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vitest/config';

export default defineConfig({
  plugins: [react()],
  // Prevent Vite from auto-discovering the repo's postcss.config.js, which
  // uses Next.js/webpack-style string plugin names ('tailwindcss') that
  // Vite's own PostCSS loader can't resolve. Tests don't need real CSS.
  css: {
    postcss: {
      plugins: [],
    },
  },
  test: {
    environment: 'jsdom',
    environmentOptions: {
      jsdom: {
        url: 'http://localhost/',
      },
    },
    globals: true,
    css: false,
    setupFiles: ['./test/vitest.setup.ts'],
    include: ['test/unit/**/*.test.{ts,tsx}', 'test/integration/**/*.test.{ts,tsx}'],
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './'),
    },
  },
});
