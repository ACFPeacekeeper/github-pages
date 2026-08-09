describe('Media Section', () => {
    it('should load the Media page', () => {
        cy.visit('http://localhost:3000/github-pages/content/media');
        cy.get('main').should('exist');
        cy.url().should('include', '/content/media');
    });
});
