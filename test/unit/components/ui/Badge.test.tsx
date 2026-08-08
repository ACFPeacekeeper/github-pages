import React from 'react';
import { render, screen } from '@testing-library/react';
import Badge from '@/src/components/ui/Badge';

describe('Badge', () => {
    it('renders children correctly', () => {
        render(<Badge>Test Badge</Badge>);
        expect(screen.getByText('Test Badge')).toBeInTheDocument();
    });

    it('applies default styles when no variant is provided', () => {
        const { container } = render(<Badge>Default</Badge>);
        const badge = container.firstChild as HTMLElement;
        expect(badge).toHaveClass('bg-blue-100');
        expect(badge).toHaveClass('px-3');
    });

    it('applies outline styles when variant is "outline"', () => {
        const { container } = render(<Badge variant="outline">Outline</Badge>);
        const badge = container.firstChild as HTMLElement;
        expect(badge).toHaveClass('border');
        expect(badge).toHaveClass('bg-transparent');
    });

    it('merges custom classNames', () => {
        const { container } = render(<Badge className="custom-class">Custom</Badge>);
        const badge = container.firstChild as HTMLElement;
        expect(badge).toHaveClass('custom-class');
    });
});
