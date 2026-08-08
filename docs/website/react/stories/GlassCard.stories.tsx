import type { Meta, StoryObj } from '@storybook/react-vite';
import GlassCard from '../site-src/components/ui/GlassCard';

const meta = {
    title: 'UI/GlassCard',
    component: GlassCard,
    tags: ['autodocs'],
} satisfies Meta<typeof GlassCard>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
    args: {
        children: (
            <div className="p-6">
                <p className="font-semibold">Glassmorphism panel</p>
                <p className="text-sm text-slate-500 dark:text-slate-400">
                    Backdrop-blur + translucent background, used for cards throughout the site.
                </p>
            </div>
        ),
    },
};
