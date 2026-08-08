import { CustomElement } from 'aurelia';
import { createSimulationController } from '../../simulations/context/createSimulationController';
import { getSimulationScenario } from '../../simulations/scenarios/scenarios';

export class ConvergenceApp {
  private readonly snapshot = createSimulationController(getSimulationScenario('balanced')).initial;
  public readonly summary = `${this.snapshot.points.length} deterministic search iterations ready to explore.`;
}

CustomElement.define({
  name: 'convergence-app',
  template: '<section aria-label="Aurelia simulation island"><h2>Optimization simulation</h2><p>${summary}</p></section>',
}, ConvergenceApp);
