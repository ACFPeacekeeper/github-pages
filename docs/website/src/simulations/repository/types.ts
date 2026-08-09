export interface SimulationScenario {
  id: string;
  name: string;
  description: string;
  seed: number;
  iterations: number;
  initialCost: number;
  convergenceRate: number;
}

export interface SimulationSample {
  iteration: number;
  cost: number;
}

export interface SimulationRun {
  scenarioId: string;
  seed: number;
  samples: SimulationSample[];
}
