import { Aurelia } from 'aurelia';
import { ConvergenceApp } from './convergence-app';

export async function mountAureliaSimulation(host: HTMLElement): Promise<() => Promise<void>> {
  const application = new Aurelia();
  application.app({ host, component: ConvergenceApp });
  await application.start();
  return async () => application.stop(true);
}
