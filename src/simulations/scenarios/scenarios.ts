import type { SimulationScenario } from '../state/types';

export const SIMULATION_SCENARIOS: SimulationScenario[] = [
  { id: 'balanced', name: 'Balanced search', description: 'A steady trade-off between exploration and refinement.', seed: 17, iterations: 28, initialCost: 148, convergenceRate: 0.09 },
  { id: 'exploratory', name: 'Exploratory search', description: 'Wider early search followed by larger incumbent improvements.', seed: 41, iterations: 28, initialCost: 164, convergenceRate: 0.075 },
  { id: 'intensified', name: 'Intensified search', description: 'Fast local gains that settle into smaller improvements.', seed: 73, iterations: 28, initialCost: 139, convergenceRate: 0.115 },
];

export function getSimulationScenario(id: string): SimulationScenario {
  return SIMULATION_SCENARIOS.find((scenario) => scenario.id === id) ?? SIMULATION_SCENARIOS[0];
}
