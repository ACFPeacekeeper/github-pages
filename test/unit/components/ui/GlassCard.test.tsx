import React from 'react';
import { render, screen } from '@testing-library/react';
import GlassCard from '@/src/components/ui/GlassCard';

describe('GlassCard', () => {
    it('renders children correctly', () => {
        render(
            <GlassCard>
                <div>Card Content</div>
            </GlassCard>
        );
        expect(screen.getByText('Card Content')).toBeInTheDocument();
    });

    it('applies default glassmorphism classes', () => {
        const { container } = render(<GlassCard>Content</GlassCard>);
        const card = container.firstChild as HTMLElement;
        expect(card).toHaveClass('backdrop-blur-md');
        expect(card).toHaveClass('bg-white/70');
    });

    it('merges custom classNames', () => {
        const { container } = render(<GlassCard className="custom-card">Content</GlassCard>);
        const card = container.firstChild as HTMLElement;
        expect(card).toHaveClass('custom-card');
    });
});
