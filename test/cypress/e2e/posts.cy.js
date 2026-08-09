describe('Posts Section', () => {
    it('should load the Posts page', () => {
        cy.visit('http://localhost:3000/github-pages/content/posts');
        cy.get('main').should('exist');
        cy.url().should('include', '/content/posts');
    });
});
