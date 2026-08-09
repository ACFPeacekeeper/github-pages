// Named .cjs (not postcss.config.js) so Docusaurus's webpack/postcss pipeline
// does not auto-discover this file. Storybook wires Tailwind via
// .storybook/main.ts viteFinal → require('./tailwind.config.cjs') + this
// plugin set when needed. Mirrors main-site postcss (tailwind + autoprefixer).
module.exports = {
  plugins: {
    tailwindcss: { config: './tailwind.config.cjs' },
    autoprefixer: {},
  },
};
