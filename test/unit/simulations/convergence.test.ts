import { describe, expect, it } from 'vitest';
import { createSimulationController } from '../../../src/simulations/context/createSimulationController';
import { generateConvergence } from '../../../src/simulations/generator/convergence';
import { getSimulationScenario } from '../../../src/simulations/scenarios/scenarios';

describe('convergence simulation', () => {
  it('generates deterministic points for a scenario', () => {
    const scenario = getSimulationScenario('balanced');
    expect(generateConvergence(scenario)).toEqual(generateConvergence(scenario));
  });

  it('never increases the incumbent solution', () => {
    const points = generateConvergence(getSimulationScenario('exploratory'));
    expect(points.every((point, index) => index === 0 || point.incumbent <= points[index - 1].incumbent)).toBe(true);
  });

  it('rejects invalid simulation parameters with scenario context', () => {
    expect(() => generateConvergence({ ...getSimulationScenario('balanced'), id: 'invalid', iterations: 1 })).toThrow(/invalid.*at least two iterations/);
  });

  it('marks the simulation complete at its final point', () => {
    const controller = createSimulationController({ ...getSimulationScenario('balanced'), iterations: 2 });
    expect(controller.advance(controller.initial).status).toBe('complete');
  });
});
