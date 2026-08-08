// Smoke tests: fast, shallow checks that the site is up and every top-level
// route renders without erroring. Deeper, content-specific assertions live
// in test/cypress/e2e/ — these just answer "is the build broken?".

const ROUTES = [
    '/',
    '/content/about',
    '/content/projects',
    '/content/reports',
    '/content/tools',
    '/content/posts',
    '/content/media',
    '/content/other',
];

describe('Smoke: site is up', () => {
    ROUTES.forEach((route) => {
        it(`loads ${route} with the layout shell intact`, () => {
            cy.visit(route);
            cy.get('header').should('exist');
            cy.get('main').should('exist');
            cy.get('footer').should('exist');
        });
    });
});

describe('Smoke: no console errors on load', () => {
    it('renders the homepage without logging a console error', () => {
        cy.visit('/', {
            onBeforeLoad(win) {
                cy.spy(win.console, 'error').as('consoleError');
            },
        });
        cy.get('main').should('exist');
        cy.get('@consoleError').should('not.have.been.called');
    });
});
