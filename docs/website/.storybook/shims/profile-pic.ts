// Stand-in for `@/assets/images/23041868.jpeg`, which Sidebar.tsx imports as
// a Next.js static image (`{ src, width, height }`). Storybook doesn't run
// through Next's image loader, so `main.ts` aliases that exact import path
// to this shim — a small inline placeholder avatar rather than the real
// (larger, licensed) photo.
const placeholder = {
    src: 'data:image/svg+xml,' +
        encodeURIComponent(
            '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">' +
            '<rect width="100" height="100" fill="#334155"/>' +
            '<circle cx="50" cy="38" r="18" fill="#94a3b8"/>' +
            '<path d="M20 90c0-20 14-32 30-32s30 12 30 32" fill="#94a3b8"/>' +
            '</svg>'
        ),
    width: 100,
    height: 100,
};

export default placeholder;
