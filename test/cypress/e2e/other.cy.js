describe('Other Section', () => {
    it('should navigate to the Other section and load the Math Curriculum', () => {
        // Visit the Other section index with absolute URL
        cy.visit('http://localhost:3000/github-pages/content/other');

        // Check if the "Mathematics Curriculum" card exists and click it
        cy.contains('Mathematics Curriculum').click();

        // Verify the URL changes to the slug
        cy.url().should('include', '/content/other/math-curriculum');

        // Verify the page title/header loads (from the injected HTML)
        cy.get('#header h2').should('contain.text', 'Curriculum');
        cy.get('#header h2').should('contain.text', 'Architecture');

        // Verify dashboard cards are present
        cy.get('#section-dashboard').should('be.visible');
        cy.contains('Total Curricular Units').should('be.visible');

        // Verify charts are rendered (canvas elements exist)
        cy.get('#subjectChart').should('exist');
        cy.get('#loadChart').should('exist');

        // Verify Sidebar/Header navigation within the split file (now Refactored to Header)
        cy.get('#nav-dashboard').should('be.visible');
        cy.get('#nav-explorer').should('be.visible');
    });

    it('should switch tabs to Course Explorer', () => {
        cy.visit('http://localhost:3000/github-pages/content/other/math-curriculum');

        // Switch to Explorer
        cy.get('#nav-explorer').click();

        // Verify Explorer section is visible
        cy.get('#section-explorer').should('not.have.class', 'hidden');
        cy.get('#section-dashboard').should('have.class', 'hidden');

        // Verify search input exists
        cy.get('#searchInput').should('be.visible');
    });
});
