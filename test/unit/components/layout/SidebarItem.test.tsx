import React from 'react';
import { render, screen } from '@testing-library/react';
import SidebarItem from '@/src/components/layout/SidebarItem';

describe('SidebarItem', () => {
    const defaultProps = {
        href: '/test',
        icon: <span data-testid="icon">Icon</span>,
        label: 'Test Item',
        active: false,
        isCollapsed: false,
    };

    it('renders correctly', () => {
        render(<SidebarItem {...defaultProps} />);
        expect(screen.getByText('Test Item')).toBeInTheDocument();
        expect(screen.getByTestId('icon')).toBeInTheDocument();
        expect(screen.getByRole('link')).toHaveAttribute('href', '/test');
    });

    it('applies active styles', () => {
        render(<SidebarItem {...defaultProps} active={true} />);
        const link = screen.getByRole('link');
        expect(link).toHaveClass('bg-blue-50'); // Checking one of the active classes
    });

    it('hides label when collapsed', () => {
        render(<SidebarItem {...defaultProps} isCollapsed={true} />);
        expect(screen.queryByText('Test Item')).not.toBeInTheDocument();
    });
});
