import React from 'react';

// See next-image.tsx for why this shim exists.
type NextLinkProps = React.AnchorHTMLAttributes<HTMLAnchorElement> & {
    href: string;
    children?: React.ReactNode;
};

export default function Link({ href, children, ...rest }: NextLinkProps) {
    return (
        <a href={href} {...rest}>
            {children}
        </a>
    );
}
