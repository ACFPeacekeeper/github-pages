import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import ReportWrapper from '@/src/frameworks/react/components/ReportWrapper';

// Mock MathJax global
beforeAll(() => {
    window.MathJax = {
        tex: { inlineMath: [], displayMath: [] },
        svg: { fontCache: '' },
        typeset: vi.fn(),
        typesetClear: vi.fn(),
    };
});

describe('ReportWrapper', () => {
    const defaultProps = {
        content: '<p>Test Report Content</p>',
        title: 'Test Report',
        styles: '.test-class { color: red; }',
        scripts: ['https://example.com/script.js'],
    };

    it('renders content safely', () => {
        render(<ReportWrapper {...defaultProps} />);
        expect(screen.getByText('Test Report Content')).toBeInTheDocument();
    });

    it('injects styles', () => {
        const { container } = render(<ReportWrapper {...defaultProps} />);
        const styleTag = container.querySelector('style');
        expect(styleTag).toBeInTheDocument();
        expect(styleTag).toHaveTextContent('.test-class { color: red; }');
    });

    it('loads external scripts', async () => {
        render(<ReportWrapper {...defaultProps} />);

        // Check if script tag is appended to body
        // Note: implementation appends to document.body
        await waitFor(() => {
            const script = document.querySelector('script[src="https://example.com/script.js"]');
            expect(script).toBeInTheDocument();
        });
    });
});
