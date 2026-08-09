/**
 * Redux store slot for docs-site demos (parity with main site libraries/redux).
 * Wire a real store when cross-island UI state is needed.
 */
export type DocsUiState = {
  theme: 'light' | 'dark';
};

export const initialDocsUiState: DocsUiState = {
  theme: 'dark',
};
