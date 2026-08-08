import React from 'react';
import { render } from '@testing-library/react';
import ClientLayoutWrapper from '@/app/ClientLayoutWrapper';

// Exercises ClientLayoutWrapper's getActiveSection logic together with the
// Sidebar/SidebarItem it feeds: a pathname change should flow through to
// the correct nav item being visually marked active. The sidebar starts
// collapsed (icon-only, no text label), so links are queried by href
// rather than accessible name.

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

describe('active section integration', () => {
    it('marks the Reports sidebar item active when the pathname is /content/reports', () => {
        currentPathname = '/content/reports';

        const { container } = render(
            <ClientLayoutWrapper>
                <div>Reports Page</div>
            </ClientLayoutWrapper>
        );

        const reportsLink = container.querySelector('aside a[href="/content/reports"]');
        expect(reportsLink).toHaveClass('bg-blue-50');
    });

    it('marks the Home sidebar item active at the root path', () => {
        currentPathname = '/';

        const { container } = render(
            <ClientLayoutWrapper>
                <div>Home Page</div>
            </ClientLayoutWrapper>
        );

        const homeLink = container.querySelector('aside a[href="/"]');
        expect(homeLink).toHaveClass('bg-blue-50');

        const reportsLink = container.querySelector('aside a[href="/content/reports"]');
        expect(reportsLink).not.toHaveClass('bg-blue-50');
    });
});
