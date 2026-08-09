/** @type {import('next').NextConfig} */
// Next.js surface for the docs website (React analogue of PMF stack/nuxt).
// The primary product is Docusaurus; this config supports multi-framework
// experiments and optional Next-based tooling under the same package.
const nextConfig = {
  basePath: '/github-pages/docs-site',
  output: 'export',
  images: {
    unoptimized: true,
  },
  env: {
    NEXT_PUBLIC_BASE_PATH: '/github-pages/docs-site',
  },
};

module.exports = nextConfig;
