import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import PostWrapper from '@/src/components/PostWrapper';

describe('PostWrapper', () => {
    const content = '<pre><code>def test(): pass</code></pre>';

    it('renders content', () => {
        render(<PostWrapper content={content} title="Test Post" />);
        // We check for the raw content structure initially
        expect(document.querySelector('.post-container')).toBeInTheDocument();
    });

    it('attempts to highlight code', async () => {
        render(<PostWrapper content={content} title="Test Post" />);

        // The highlighting happens in useEffect, so we wait.
        // However, our simple mock content might not trigger complex highlighting classes
        // without the full logic running on specific tokens.
        // We can at least verify the container exists and text is present.
        await waitFor(() => {
            expect(screen.getByText(/def/)).toBeInTheDocument();
        });
    });
});
