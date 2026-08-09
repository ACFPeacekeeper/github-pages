import React from 'react';
import type { Preview } from '@storybook/react-vite';
import './tailwind-entry.css';

// Components style themselves via Tailwind's `dark:` variant off a `dark`
// class on an ancestor (tailwind.config.cjs: darkMode: 'class') — the
// toolbar toggle below drives that class here the same way
// app/ClientLayoutWrapper.tsx drives it on <html> in the real site.
const preview: Preview = {
    parameters: {
        controls: {
            matchers: {
                color: /(background|color)$/i,
                date: /Date$/i,
            },
        },
        backgrounds: {
            default: 'dark',
            values: [
                { name: 'dark', value: '#0f172a' },
                { name: 'light', value: '#f8fafc' },
            ],
        },
    },
    globalTypes: {
        theme: {
            description: 'Light/dark theme',
            defaultValue: 'dark',
            toolbar: {
                title: 'Theme',
                icon: 'mirror',
                items: [
                    { value: 'light', title: 'Light' },
                    { value: 'dark', title: 'Dark' },
                ],
                dynamicTitle: true,
            },
        },
    },
    decorators: [
        (Story, context) => {
            const theme = context.globals.theme ?? 'dark';
            return (
                <div className={theme === 'dark' ? 'dark' : ''}>
                    <div className="bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-200 p-6 min-h-[100px]">
                        <Story />
                    </div>
                </div>
            );
        },
    ],
};

export default preview;
