import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import ResearchConstellation from '../../../../src/components/graph/ResearchConstellation';

describe('ResearchConstellation', () => {
  it('describes the initially selected research node', () => {
    render(<ResearchConstellation />);
    expect(screen.getByText('The bridge between mathematical ideas and useful systems.')).toBeInTheDocument();
  });

  it('updates the explanation when a visitor selects a node', async () => {
    const user = userEvent.setup();
    render(<ResearchConstellation />);
    await user.click(screen.getByRole('button', { name: /Artificial intelligence:/ }));
    expect(screen.getByText('Learning representations, policies, and useful heuristics from data.')).toBeInTheDocument();
  });

  it('exposes a relevant destination for linked nodes', async () => {
    const user = userEvent.setup();
    render(<ResearchConstellation />);
    await user.click(screen.getByRole('button', { name: /Vehicle routing:/ }));
    expect(screen.getByRole('link', { name: /Explore this thread/ })).toHaveAttribute('href', '/github-pages/content/posts/Attention_Learn_to_Solve_Routing_Problem');
  });
});
