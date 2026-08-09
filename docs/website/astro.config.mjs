import { defineConfig } from 'astro/config';

// Docs-site Astro island → static/astro-island (served by Docusaurus static/).
export default defineConfig({
  srcDir: './src/frameworks/astro',
  outDir: './static/astro-island',
  publicDir: './astro-public',
  base: '/github-pages/docs-site/astro-island',
  build: {
    format: 'directory',
  },
});
