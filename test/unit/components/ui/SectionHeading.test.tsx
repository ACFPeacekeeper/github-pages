import React from 'react';
import { render, screen } from '@testing-library/react';
import SectionHeading from '@/src/components/ui/SectionHeading';

describe('SectionHeading', () => {
    const mockIcon = <span data-testid="mock-icon">Icon</span>;

    it('renders title and icon correctly', () => {
        render(<SectionHeading title="Test Title" icon={mockIcon} />);

        expect(screen.getByText('Test Title')).toBeVisible();
        expect(screen.getByTestId('mock-icon')).toBeInTheDocument();
    });

    it('applies custom classNames', () => {
        const { container } = render(<SectionHeading title="Title" icon={mockIcon} className="custom-heading" />);
        const heading = container.firstChild as HTMLElement;
        expect(heading).toHaveClass('custom-heading');
    });
});
