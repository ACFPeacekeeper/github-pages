import type { Meta, StoryObj } from '@storybook/react-vite';
import { fn } from 'storybook/test';
import Footer from '../../../src/frameworks/react/components/layout/Footer';

const meta = {
    title: 'Layout/Footer',
    component: Footer,
    tags: ['autodocs'],
    args: {
        toggleTheme: fn(),
    },
    argTypes: {
        darkMode: { control: 'boolean', description: 'Whether dark mode is active.' },
        toggleTheme: { description: 'Called when the theme toggle button is clicked.' },
    },
} satisfies Meta<typeof Footer>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Light: Story = {
    args: { darkMode: false },
};

export const Dark: Story = {
    args: { darkMode: true },
};
