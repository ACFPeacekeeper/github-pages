// `npm run typecheck` (plain `tsc`, not part of the actual Docusaurus/Vite
// build) needs to know the shape of static image imports, since the main
// repo's own ambient types (next-env.d.ts) aren't part of this project.
// Matches Next.js's StaticImageData shape (not a plain string, unlike
// Vite's own default asset typing) since site-src/ is the real Next.js
// component source, written against that contract.
interface StaticImageData {
    src: string;
    height: number;
    width: number;
    blurDataURL?: string;
}

declare module '*.jpeg' {
    const data: StaticImageData;
    export default data;
}
declare module '*.jpg' {
    const data: StaticImageData;
    export default data;
}
declare module '*.png' {
    const data: StaticImageData;
    export default data;
}
declare module '*.svg' {
    const data: StaticImageData;
    export default data;
}
