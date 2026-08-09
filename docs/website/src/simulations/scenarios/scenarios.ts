import type { SimulationScenario } from '../repository/types';

export const SIMULATION_SCENARIOS: SimulationScenario[] = [
  {
    id: 'balanced',
    name: 'Balanced search',
    description: 'A steady trade-off between exploration and refinement.',
    seed: 17,
    iterations: 28,
    initialCost: 148,
    convergenceRate: 0.09,
  },
  {
    id: 'exploratory',
    name: 'Exploratory search',
    description: 'Wider early search followed by larger incumbent improvements.',
    seed: 41,
    iterations: 28,
    initialCost: 164,
    convergenceRate: 0.075,
  },
];

export function getSimulationScenario(id: string): SimulationScenario {
  return SIMULATION_SCENARIOS.find((s) => s.id === id) ?? SIMULATION_SCENARIOS[0];
}
