import { setupServer } from 'msw/node';
import { handlers } from './handlers';

/**
 * Node-side MSW server for integration tests. Started/stopped in
 * test/vitest.setup.ts. `onUnhandledRequest: 'error'` means any test that
 * accidentally reaches the real network fails loudly instead of silently
 * hitting a live endpoint.
 */
export const server = setupServer(...handlers);
