import type { Meta, StoryObj } from '@storybook/react-vite';
import { Home } from 'lucide-react';
import SidebarItem from '../../../src/frameworks/react/components/layout/SidebarItem';

const meta = {
    title: 'Layout/SidebarItem',
    component: SidebarItem,
    tags: ['autodocs'],
    argTypes: {
        href: { control: 'text', description: 'Link target, passed straight through to next/link.' },
        label: { control: 'text', description: 'Visible label; hidden entirely when isCollapsed.' },
        active: { control: 'boolean', description: 'Whether this item represents the current route.' },
        isCollapsed: { control: 'boolean', description: 'Icon-only rendering for the collapsed sidebar rail.' },
    },
} satisfies Meta<typeof SidebarItem>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Inactive: Story = {
    args: {
        href: '/content/about',
        icon: <Home size={20} />,
        label: 'About',
        active: false,
        isCollapsed: false,
    },
};

export const Active: Story = {
    args: { ...Inactive.args, active: true },
};

export const Collapsed: Story = {
    args: { ...Inactive.args, isCollapsed: true },
};
