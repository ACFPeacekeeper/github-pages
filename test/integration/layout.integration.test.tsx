import React from 'react';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import ClientLayoutWrapper from '@/app/ClientLayoutWrapper';

// ClientLayoutWrapper composes Header + Sidebar + Footer; this suite
// exercises the real integration between them (shared theme state,
// pathname-driven active section) rather than any one in isolation.

let currentPathname = '/';
vi.mock('next/navigation', () => ({
    usePathname: () => currentPathname,
}));

vi.mock('next/image', () => ({
    __esModule: true,
    default: (props: any) => <img {...props} />,
}));

vi.mock('@/assets/images/23041868.jpeg', () => ({
    default: { src: '/mock-image.jpg', height: 100, width: 100 },
}));

describe('ClientLayoutWrapper integration', () => {
    beforeEach(() => {
        currentPathname = '/';
        window.localStorage.clear();
    });

    it('renders the Header, Sidebar, and Footer together with the page content', () => {
        render(
            <ClientLayoutWrapper>
                <div>Page Content</div>
            </ClientLayoutWrapper>
        );

        // Header logo, Sidebar nav item, Footer copyright, and children all present.
        expect(screen.getAllByText('ACF').length).toBeGreaterThan(0);
        expect(screen.getByText('Page Content')).toBeInTheDocument();
        expect(screen.getByText(/All rights reserved/)).toBeInTheDocument();
    });

    it('toggling the theme in the Header updates the Footer icon and persists to localStorage', async () => {
        const user = userEvent.setup();
        render(
            <ClientLayoutWrapper>
                <div>Page Content</div>
            </ClientLayoutWrapper>
        );

        // Starts dark by default (no stored theme).
        expect(window.localStorage.getItem('theme')).toBe('dark');

        const headerToggle = screen.getByLabelText('Toggle theme');
        await user.click(headerToggle);

        expect(window.localStorage.getItem('theme')).toBe('light');
    });

    it('reads a previously stored light theme on mount', () => {
        window.localStorage.setItem('theme', 'light');

        render(
            <ClientLayoutWrapper>
                <div>Page Content</div>
            </ClientLayoutWrapper>
        );

        expect(document.documentElement.classList.contains('dark')).toBe(false);
    });
});
