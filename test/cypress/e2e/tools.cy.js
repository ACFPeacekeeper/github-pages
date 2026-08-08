describe('Tools Section', () => {
    it('should load the Tools page', () => {
        cy.visit('http://localhost:3000/github-pages/content/tools');
        cy.get('main').should('exist');
        cy.url().should('include', '/content/tools');
    });
});
