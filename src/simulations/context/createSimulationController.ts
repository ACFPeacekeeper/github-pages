import { generateConvergence } from '../generator/convergence';
import type { SimulationScenario, SimulationSnapshot } from '../repository/types';

export interface SimulationController {
  initial: SimulationSnapshot;
  advance(snapshot: SimulationSnapshot): SimulationSnapshot;
  reset(): SimulationSnapshot;
}

export function createSimulationController(scenario: SimulationScenario): SimulationController {
  const points = generateConvergence(scenario);
  const reset = (): SimulationSnapshot => ({ status: 'idle', cursor: 0, points });
  return {
    initial: reset(),
    advance(snapshot) {
      const cursor = Math.min(snapshot.cursor + 1, points.length - 1);
      return { ...snapshot, cursor, status: cursor === points.length - 1 ? 'complete' : 'running' };
    },
    reset,
  };
}
