import type { Meta, StoryObj } from '@storybook/react-vite';
import { fn } from 'storybook/test';
import Header from '../site-src/components/layout/Header';

// Header calls next/navigation's usePathname(), which the Vite alias in
// .storybook/main.ts points at .storybook/shims/next-navigation.ts — it
// reads window.__STORYBOOK_PATHNAME__, set here per story via `beforeEach`.
const meta = {
    title: 'Layout/Header',
    component: Header,
    tags: ['autodocs'],
    args: {
        toggleTheme: fn(),
    },
    argTypes: {
        darkMode: { control: 'boolean', description: 'Whether dark mode is active.' },
        toggleTheme: { description: 'Called when the theme toggle button is clicked.' },
    },
} satisfies Meta<typeof Header>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
    args: { darkMode: false },
    beforeEach: () => {
        (globalThis as { __STORYBOOK_PATHNAME__?: string }).__STORYBOOK_PATHNAME__ = '/';
    },
};

export const OnReportsPage: Story = {
    args: { darkMode: false },
    beforeEach: () => {
        (globalThis as { __STORYBOOK_PATHNAME__?: string }).__STORYBOOK_PATHNAME__ = '/content/reports';
    },
};

export const Dark: Story = {
    args: { darkMode: true },
    beforeEach: () => {
        (globalThis as { __STORYBOOK_PATHNAME__?: string }).__STORYBOOK_PATHNAME__ = '/';
    },
};
