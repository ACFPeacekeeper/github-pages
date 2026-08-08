const { defineConfig } = require("cypress");

module.exports = defineConfig({
    e2e: {
        baseUrl: 'http://localhost:3000/github-pages',
        specPattern: ['e2e/**/*.cy.js', 'smoke/**/*.cy.js'],
        screenshotsFolder: 'screenshots',
        videosFolder: 'videos',
        setupNodeEvents(on, config) {
            // implement node event listeners here
        },
        supportFile: false,
    },
});
