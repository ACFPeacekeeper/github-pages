describe('Navigation', () => {
    it('should navigate to valid pages from the Sidebar/Header', () => {
        cy.visit('http://localhost:3000/github-pages/');

        // Check for "Other" link in the main navigation
        // Note: Adjust selector based on actual generic sidebar/header implementation
        // We expect an anchor with href="/content/other"
        cy.get('a[href="/content/other"]').click();

        // Verify we are on the correct page
        cy.url().should('include', '/content/other');
        cy.contains('Other Projects').should('be.visible');
    });

    it('should display the Math Curriculum card on the Other page', () => {
        cy.visit('http://localhost:3000/github-pages/content/other');
        cy.contains('Mathematics Curriculum').should('be.visible');
    });
});
