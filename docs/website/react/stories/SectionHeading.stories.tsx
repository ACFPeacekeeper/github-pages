import type { Meta, StoryObj } from '@storybook/react-vite';
import { FileText } from 'lucide-react';
import SectionHeading from '../../../../src/components/ui/SectionHeading';

const meta = {
    title: 'UI/SectionHeading',
    component: SectionHeading,
    tags: ['autodocs'],
} satisfies Meta<typeof SectionHeading>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
    args: {
        title: 'Reports',
        icon: <FileText size={20} />,
    },
};
