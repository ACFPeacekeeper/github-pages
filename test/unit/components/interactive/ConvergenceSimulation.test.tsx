import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import ConvergenceSimulation from '../../../../src/components/interactive/ConvergenceSimulation';

describe('ConvergenceSimulation', () => {
  it('exposes the initial metrics as accessible text', () => {
    render(<ConvergenceSimulation />);
    expect(screen.getByText('1/28')).toBeInTheDocument();
    expect(screen.getByRole('img', { name: /Optimization convergence chart/ })).toHaveAccessibleDescription(/iteration 0/);
  });

  it('resets the selected scenario to its first iteration', async () => {
    const user = userEvent.setup();
    render(<ConvergenceSimulation />);
    await user.selectOptions(screen.getByLabelText('Search strategy'), 'intensified');
    await user.click(screen.getByRole('button', { name: 'Run simulation' }));
    await user.click(screen.getByRole('button', { name: /Reset/ }));
    expect(screen.getByText('1/28')).toBeInTheDocument();
  });
});
