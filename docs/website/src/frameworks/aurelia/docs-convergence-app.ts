import { CustomElement } from 'aurelia';
import { createSimulationController } from '../../simulations/context/createSimulationController';
import { getSimulationScenario } from '../../simulations/scenarios/scenarios';

/** Aurelia custom element showcasing the docs-site simulation controller. */
export class DocsConvergenceApp {
  private readonly controller = createSimulationController(
    getSimulationScenario('balanced').id
  );
  public readonly summary = `${this.controller.run.samples.length} deterministic search iterations ready to explore.`;
  public readonly scenarioName = this.controller.scenarioId;
}

CustomElement.define(
  {
    name: 'docs-convergence-app',
    template:
      '<section aria-label="Aurelia simulation island" class="aurelia-island">' +
      '<p class="kicker">Framework island · Aurelia</p>' +
      '<h2>Optimization simulation</h2>' +
      '<p>${summary}</p>' +
      '<p class="meta">Scenario: ${scenarioName}</p>' +
      '</section>',
  },
  DocsConvergenceApp
);
