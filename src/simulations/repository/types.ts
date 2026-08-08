export type SimulationStatus = 'idle' | 'running' | 'complete';

export interface SimulationScenario {
  id: string;
  name: string;
  description: string;
  seed: number;
  iterations: number;
  initialCost: number;
  convergenceRate: number;
}

export interface SimulationPoint {
  iteration: number;
  incumbent: number;
  lowerBound: number;
}

export interface SimulationSnapshot {
  status: SimulationStatus;
  cursor: number;
  points: SimulationPoint[];
}
