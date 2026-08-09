import '@testing-library/jest-dom';
import { afterAll, afterEach, beforeAll, beforeEach } from 'vitest';
import { server } from './integration/mocks/server';

beforeAll(() => server.listen({ onUnhandledRequest: 'error' }));
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

// Node's own built-in `localStorage` global (stable since Node 22) shadows
// jsdom's window.localStorage in a way that leaves it non-functional
// without a `--localstorage-file` path. Replace it with a plain in-memory
// implementation so ClientLayoutWrapper's theme persistence is testable.
class MemoryStorage implements Storage {
    private store = new Map<string, string>();

    get length(): number {
        return this.store.size;
    }

    clear(): void {
        this.store.clear();
    }

    getItem(key: string): string | null {
        return this.store.has(key) ? this.store.get(key)! : null;
    }

    key(index: number): string | null {
        return Array.from(this.store.keys())[index] ?? null;
    }

    removeItem(key: string): void {
        this.store.delete(key);
    }

    setItem(key: string, value: string): void {
        this.store.set(key, String(value));
    }
}

const memoryStorage = new MemoryStorage();
Object.defineProperty(window, 'localStorage', { value: memoryStorage, writable: true });
Object.defineProperty(globalThis, 'localStorage', { value: memoryStorage, writable: true });

beforeEach(() => memoryStorage.clear());
