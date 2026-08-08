import type { Meta, StoryObj } from '@storybook/react-vite';
import Badge from '../../../../src/components/ui/Badge';

const meta = {
    title: 'UI/Badge',
    component: Badge,
    tags: ['autodocs'],
    argTypes: {
        variant: {
            control: 'select',
            options: ['default', 'outline'],
        },
    },
} satisfies Meta<typeof Badge>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
    args: {
        children: 'New',
    },
};

export const Outline: Story = {
    args: {
        children: 'Draft',
        variant: 'outline',
    },
};
