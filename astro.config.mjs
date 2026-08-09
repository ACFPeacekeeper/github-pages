import { defineConfig } from 'astro/config';

// Astro sources live under src/frameworks/astro (pages/, components/, *.astro).
// Output is a static island consumed by the Next host via iframe.
export default defineConfig({
  srcDir: './src/frameworks/astro',
  outDir: './public/astro-island',
  publicDir: './astro-public',
  base: '/github-pages/astro-island',
  vite: {
    css: {
      postcss: {},
    },
  },
});
