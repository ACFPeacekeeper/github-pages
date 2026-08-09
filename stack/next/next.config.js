/** @type {import('next').NextConfig} */
const nextConfig = {
  // Sets the base path for deployment to a subdirectory (e.g., GitHub Pages)
  basePath: '/github-pages',
  
  // Required for static exports (e.g., for GitHub Pages deployment)
  output: 'export', 
  
  // For handling image imports and paths correctly in static exports
  images: {
    unoptimized: true,
  },

  // Explicitly set the client-side environment variable to guarantee
  // that the component can access the basePath on the client.
  env: {
    NEXT_PUBLIC_BASE_PATH: '/github-pages',
  },
};

module.exports = nextConfig;