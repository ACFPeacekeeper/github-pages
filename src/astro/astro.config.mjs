import { defineConfig } from 'astro/config';

export default defineConfig({
  srcDir: './src',
  outDir: '../../public/astro-island',
  publicDir: './public',
  base: '/github-pages/astro-island',
  vite: {
    css: {
      postcss: {}
    }
  }
});
