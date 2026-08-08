import { Aurelia } from 'aurelia';
import { HelloAurelia } from './HelloAurelia';

export async function mountHelloAurelia(host: HTMLElement): Promise<() => Promise<void>> {
  const application = new Aurelia();
  application.app({ host, component: HelloAurelia });
  await application.start();
  return async () => application.stop(true);
}
