// Named .cjs (not tailwind.config.js/postcss.config.*) so it is never
// auto-discovered by Docusaurus's own webpack/postcss pipeline — only
// .storybook/main.ts's viteFinal wires this in explicitly, via Vite's
// inline `css.postcss` option rather than a root-level postcss.config.js.
// That keeps Tailwind's preflight reset scoped to Storybook and away from
// Docusaurus's Infima-based site styling, which the two would otherwise
// silently fight over if both builds ran from this same directory.
/** @type {import('tailwindcss').Config} */
module.exports = {
    content: ['../../src/**/*.{js,jsx,ts,tsx}'],
    darkMode: 'class',
    theme: {
        extend: {
            fontFamily: {
                sans: ['Inter', 'sans-serif'],
                display: ['Lexend', 'sans-serif'],
            },
            colors: {
                slate: {
                    850: '#1e293b',
                    900: '#0f172a',
                },
                primary: {
                    500: '#3b82f6',
                    600: '#2563eb',
                },
            },
        },
    },
    plugins: [],
};
