export function logIslandMount(framework: string, elementId?: string) {
  console.log(`[Island Architecture] Mounted ${framework} island${elementId ? ` at #${elementId}` : ''}.`);
}

export function generateIslandId(prefix: string): string {
  return `${prefix}-${Math.random().toString(36).substring(2, 9)}`;
}
