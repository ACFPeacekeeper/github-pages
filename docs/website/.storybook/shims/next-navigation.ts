// See next-image.tsx for why this shim exists. Stories that care about a
// specific active route can override this via the `pathname` field on
// window.__STORYBOOK_PATHNAME__ before render (see Header.stories.tsx for
// an example); everything else gets a sensible default of "/".
export function usePathname(): string {
    return (globalThis as { __STORYBOOK_PATHNAME__?: string }).__STORYBOOK_PATHNAME__ ?? '/';
}

export function notFound(): never {
    throw new Error('next/navigation notFound() called in Storybook — this shim does not render a 404 page.');
}
