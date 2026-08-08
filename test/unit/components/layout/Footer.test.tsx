import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import Footer from '@/src/components/layout/Footer';

describe('Footer', () => {
    const mockToggleTheme = vi.fn();

    it('renders copyright and links', () => {
        render(<Footer darkMode={false} toggleTheme={mockToggleTheme} />);
        expect(screen.getByText(/All rights reserved/)).toBeInTheDocument();
        expect(screen.getByText('RSS')).toBeInTheDocument();
    });

    it('toggles theme when button is clicked', () => {
        render(<Footer darkMode={false} toggleTheme={mockToggleTheme} />);
        const button = screen.getByLabelText('Toggle Dark Mode');
        fireEvent.click(button);
        expect(mockToggleTheme).toHaveBeenCalledTimes(1);
    });
});
