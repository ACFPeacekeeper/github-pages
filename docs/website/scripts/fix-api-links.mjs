#!/usr/bin/env node
// Runs after `typedoc` (see package.json's "gen:api" script). TypeDoc's
// generated markdown cross-links other generated pages with a relative,
// ".md"-suffixed path (e.g. "[getAllPageSlugs](functions/getAllPageSlugs.md)").
// Docusaurus's relative-doc-link resolver mis-threads the webpack import for
// that link's target through the wrong docs-plugin instance's generated
// content folder ("Cannot find module '@site/.docusaurus/.../default/...'"
// for a file that actually exists under a differently-named plugin folder)
// — rewriting them as absolute site paths sidesteps that resolver
// (Docusaurus only applies it to relative links) rather than fighting it.

import { readdirSync, readFileSync, writeFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const apiDocsDir = path.resolve(__dirname, '../api-docs');

// Must match the "default" docs-plugin's routeBasePath ("docs") + this
// directory's position under its `path` (docs/) in docusaurus.config.ts.
const API_DOCS_BASE_ROUTE = '/docs/website/api-docs';

function walk(dir) {
    for (const entry of readdirSync(dir)) {
        const fullPath = path.join(dir, entry);
        if (statSync(fullPath).isDirectory()) {
            walk(fullPath);
        } else if (entry.endsWith('.md')) {
            const original = readFileSync(fullPath, 'utf8');
            const fixed = original.replace(
                /\]\(([^)#\s]+)\.md(#[^)]*)?\)/g,
                (_match, relativePath, hash = '') => `](${API_DOCS_BASE_ROUTE}/${relativePath}${hash})`
            );
            if (fixed !== original) {
                writeFileSync(fullPath, fixed);
            }
        }
    }
}

walk(apiDocsDir);
console.log('fix-api-links: rewrote api-docs/ internal links as absolute site paths');
