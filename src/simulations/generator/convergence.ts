import type { SimulationPoint, SimulationScenario } from '../state/types';

function createRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

export function generateConvergence(scenario: SimulationScenario): SimulationPoint[] {
  if (scenario.iterations < 2 || scenario.initialCost <= 0 || scenario.convergenceRate <= 0) {
    throw new Error(`Cannot generate simulation for scenario "${scenario.id}": parameters must be positive and include at least two iterations.`);
  }

  const random = createRandom(scenario.seed);
  let incumbent = scenario.initialCost;
  return Array.from({ length: scenario.iterations }, (_, iteration) => {
    const theoretical = scenario.initialCost * (0.52 + 0.48 * Math.exp(-scenario.convergenceRate * iteration));
    if (iteration > 0 && random() > 0.38) incumbent = Math.min(incumbent, theoretical + random() * 4);
    const lowerBound = scenario.initialCost * (0.42 + 0.35 * (1 - Math.exp(-scenario.convergenceRate * iteration * 0.7)));
    return { iteration, incumbent: Number(incumbent.toFixed(1)), lowerBound: Number(Math.min(lowerBound, incumbent).toFixed(1)) };
  });
}
