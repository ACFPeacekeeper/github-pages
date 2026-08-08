describe('About Section', () => {
    it('should load the About page', () => {
        cy.visit('http://localhost:3000/github-pages/content/about');
        // Verify common layout elements or specific about content exist
        cy.get('main').should('exist');
        // Adjust selector based on actual content, assuming generic header presence
        cy.url().should('include', '/content/about');
    });
});
