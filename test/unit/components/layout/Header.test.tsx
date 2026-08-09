import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import Header from '@/src/frameworks/react/components/layout/Header';

// Mock usePathname
vi.mock('next/navigation', () => ({
    usePathname: () => '/',
}));

describe('Header', () => {
    const mockToggleTheme = vi.fn();

    it('renders logo and navigation items', () => {
        render(<Header darkMode={false} toggleTheme={mockToggleTheme} />);
        expect(screen.getByText('ACF')).toBeInTheDocument();
        expect(screen.getByText('Harbinger')).toBeInTheDocument();
    });

    it('toggles mobile menu', () => {
        render(<Header darkMode={false} toggleTheme={mockToggleTheme} />);

        // Initially menu should be hidden (or button present)
        // Using getByRole is more robust and accessible
        const menuButton = screen.getByRole('button', { name: /open main menu/i });
        fireEvent.click(menuButton);

        // Check if menu items are visible
        expect(screen.getAllByText('Home')[0]).toBeInTheDocument();
    });
});
