describe('Smoke: theme toggle', () => {
    it('defaults to dark mode and switches to light on click', () => {
        cy.visit('/');
        cy.get('html').should('have.class', 'dark');

        cy.get('button[aria-label="Toggle theme"]').first().click();
        cy.get('html').should('not.have.class', 'dark');
    });
});
