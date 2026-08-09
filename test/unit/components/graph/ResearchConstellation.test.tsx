import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import ResearchConstellation from '../../../../src/frameworks/react/components/graph/ResearchConstellation';

describe('ResearchConstellation', () => {
  it('describes the initially selected research node', () => {
    render(<ResearchConstellation />);
    expect(screen.getAllByText('The bridge between mathematical ideas and useful systems.')[0]).toBeInTheDocument();
  });

  it('updates the explanation when a visitor selects a node', async () => {
    const user = userEvent.setup();
    render(<ResearchConstellation />);
    await user.click(screen.getByRole('button', { name: /Artificial intelligence:/ }));
    expect(screen.getAllByText('Learning representations, policies, and useful heuristics from data.')[0]).toBeInTheDocument();
  });

  it('exposes a relevant destination for linked nodes', async () => {
    const user = userEvent.setup();
    render(<ResearchConstellation />);
    await user.click(screen.getByRole('button', { name: /Vehicle routing:/ }));
    expect(screen.getByRole('link', { name: /Explore this thread/ })).toHaveAttribute('href', '/github-pages/content/posts/Attention_Learn_to_Solve_Routing_Problem');
  });
});
