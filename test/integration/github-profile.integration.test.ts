import { http, HttpResponse } from 'msw';
import { server } from './mocks/server';

// The Sidebar links out to https://github.com/acfharbinger but doesn't
// fetch anything today. This suite exercises the MSW-backed network layer
// integration tests run against, using that same endpoint as a realistic
// stand-in: it proves the default handler resolves, that a test can
// override it for a single case, and that truly unhandled requests are
// rejected rather than silently hitting the real network.

describe('MSW-backed network layer', () => {
    it('resolves the default handler for the linked GitHub profile', async () => {
        const response = await fetch('https://api.github.com/users/acfharbinger');
        const data = await response.json();

        expect(response.status).toBe(200);
        expect(data.login).toBe('acfharbinger');
    });

    it('allows a test to override a handler for one request', async () => {
        server.use(
            http.get('https://api.github.com/users/acfharbinger', () => {
                return HttpResponse.json({ login: 'acfharbinger', followers: 1000 }, { status: 200 });
            })
        );

        const response = await fetch('https://api.github.com/users/acfharbinger');
        const data = await response.json();

        expect(data.followers).toBe(1000);
    });

    it('rejects requests with no matching handler instead of hitting the real network', async () => {
        await expect(fetch('https://api.github.com/users/does-not-exist-in-handlers')).rejects.toThrow();
    });
});
