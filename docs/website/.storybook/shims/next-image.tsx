import React from 'react';

// Storybook here runs on plain Vite (@storybook/react-vite), not Next.js —
// this repo's components are the real ../../../src/components/ sources, so
// their `next/*` imports need a lightweight stand-in rather than pulling in
// the full Next.js runtime. See ../main.ts's `resolve.alias`.
type NextImageProps = React.ImgHTMLAttributes<HTMLImageElement> & {
    src: string | { src: string };
};

export default function Image({ src, alt = '', ...rest }: NextImageProps) {
    const resolvedSrc = typeof src === 'string' ? src : src.src;
    // eslint-disable-next-line @next/next/no-img-element
    return <img src={resolvedSrc} alt={alt} {...rest} />;
}
