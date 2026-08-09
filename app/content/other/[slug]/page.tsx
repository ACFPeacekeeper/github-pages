import React from 'react';
import { notFound } from 'next/navigation';
import { ArrowLeft } from 'lucide-react';
import Link from 'next/link';
import fs from 'fs';
import path from 'path';
import ReportWrapper from '@/src/frameworks/react/ReportWrapper';

// Configuration: Map the slug to the Base Filename (no extension) and Title
const OTHER_DATA: Record<string, { baseName: string; title: string }> = {
    'math-curriculum': {
        baseName: 'Mathematics_Curriculum',
        title: 'Mathematics Curriculum'
    },
};

export async function generateStaticParams() {
    return Object.keys(OTHER_DATA).map((slug) => ({
        slug: slug,
    }));
}

interface PageProps {
    params: { slug: string };
}

/**
 * Loads and processes the assets from the new directory structure:
 * - HTML: /public/other/html/[baseName].html
 * - CSS:  /public/other/css/[baseName].css
 * - JS:   /public/other/javascript/[baseName].js
 */
function loadOtherAssets(baseName: string) {
    const basePath = path.join(process.cwd(), 'public', 'other');

    // Paths
    const htmlPath = path.join(basePath, 'html', `${baseName}.html`);
    const cssPath = path.join(basePath, 'css', `${baseName}.css`);
    const jsPath = path.join(basePath, 'javascript', `${baseName}.js`);

    let bodyContent = '';
    let styleContent = '';
    let inlineScriptContent = '';

    // 1. Load HTML (Required)
    try {
        const rawHtml = fs.readFileSync(htmlPath, 'utf8');

        // Clean up HTML: Extract content inside <body> or return raw if no body tag
        const bodyMatch = rawHtml.match(/<body[^>]*>([\s\S]*?)<\/body>/i);
        bodyContent = bodyMatch ? bodyMatch[1] : rawHtml;

        // Remove any leftover <script> or <style> tags from the HTML to be safe
        bodyContent = bodyContent.replace(/<script[^>]*>[\s\S]*?<\/script>/gi, '');
        bodyContent = bodyContent.replace(/<style[^>]*>[\s\S]*?<\/style>/gi, '');

    } catch (err) {
        console.error(`HTML file not found for: ${baseName}`, err);
        throw new Error('Project HTML missing');
    }

    // 2. Load CSS (Optional)
    try {
        if (fs.existsSync(cssPath)) {
            let rawCss = fs.readFileSync(cssPath, 'utf8');

            // Scope the CSS to .report-content to prevent global leaks
            rawCss = rawCss.replace(/(^|[\s,}])\b(body|html)\b(?=[\s,{])/gi, '$1.report-content');
            styleContent = rawCss;
        }
    } catch (err) {
        console.warn(`CSS file not found for: ${baseName}`);
    }

    // 3. Load Javascript (Optional)
    try {
        if (fs.existsSync(jsPath)) {
            let rawJs = fs.readFileSync(jsPath, 'utf8');

            // B. Handle DOMContentLoaded Logic
            // ---------------------------------------------------------
            // Dynamically select regex based on file structure.
            // 1. Lazy Match (Default): Stops at first "});". Safe for simple files.
            // 2. Greedy Match (Complex): Consumes everything until the last "});". Required for files with nested "});" inside the listener (e.g. Chart.js configs).
            // ---------------------------------------------------------
            let domLoadedRegex = /document\.addEventListener\s*\(\s*['"]DOMContentLoaded['"]\s*,\s*(\(\)\s*=>\s*|function\s*)\s*{([\s\S]*?)}\s*\)\s*;/i;

            const initMatch = rawJs.match(domLoadedRegex);

            let initCode = '';
            if (initMatch) {
                initCode = initMatch[2].trim();
                // Remove the listener block from the main script to avoid duplication
                rawJs = rawJs.replace(domLoadedRegex, '');
            }

            // C. Expose Functions to Window
            // Use simple direct assignment with guaranteed termination to prevent SyntaxError.
            const functionMatches = Array.from(rawJs.matchAll(/function\s+([a-zA-Z0-9_]+)\s*\(/g));
            let functionExports = '';

            for (const match of functionMatches) {
                const funcName = match[1];
                // Direct assignment with semicolon termination
                functionExports += `\nif (typeof ${funcName} !== 'undefined') { window.${funcName} = ${funcName}; };`;
            }

            // D. Combine: 
            // 1. Original declarations (safe)
            // 2. Window assignments (guaranteed semicolon termination)
            // 3. Initialization code wrapped in a block {} to prevent 'const' redeclaration errors
            inlineScriptContent = `${rawJs}\n\n// --- Exports ---\n${functionExports}\n\n// --- Initialization ---\n{\n${initCode}\n}`;
        }
    } catch (err) {
        console.warn(`JS file not found for: ${baseName}`);
    }

    return { bodyContent, styleContent, inlineScriptContent };
}

export default function OtherPage({ params }: PageProps) {
    const { slug } = params;
    const projectConfig = OTHER_DATA[slug];

    if (!projectConfig) {
        return notFound();
    }

    let parsedData = { bodyContent: '', styleContent: '', inlineScriptContent: '' };

    // Script dependencies (like Chart.js + Tailwind for this project)
    const scriptFiles = [
        'https://cdn.jsdelivr.net/npm/chart.js',
        'https://cdn.tailwindcss.com'
    ];

    try {
        parsedData = loadOtherAssets(projectConfig.baseName);
    } catch (error) {
        console.error(`Error loading assets for ${projectConfig.baseName}:`, error);
        notFound();
    }

    return (
        <div className="animate-in fade-in duration-500 flex flex-col min-h-screen bg-slate-50 dark:bg-slate-950">

            <main className="flex-1 w-full pb-12">
                <div className="max-w-7xl mx-auto w-full px-4 sm:px-6 lg:px-8">
                    <div className="mb-6 pt-10">
                        <Link
                            href="/content/other"
                            className="flex items-center text-slate-600 dark:text-slate-400 hover:text-blue-500 transition-colors w-fit group"
                        >
                            <ArrowLeft size={20} className="mr-2 group-hover:-translate-x-1 transition-transform" />
                            Back to Projects List
                        </Link>
                    </div>

                    <h1 className="text-3xl font-bold mb-6 capitalize text-slate-900 dark:text-white">
                        {projectConfig.title}
                    </h1>

                    <ReportWrapper
                        content={parsedData.bodyContent}
                        styles={parsedData.styleContent}
                        scripts={scriptFiles}
                        inlineScript={parsedData.inlineScriptContent}
                        title={projectConfig.title}
                    />
                </div>
            </main>
        </div>
    );
}
