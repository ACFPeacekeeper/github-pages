import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import Sidebar from '@/src/components/layout/Sidebar';

// Mock next/image since Sidebar uses import profilePic
vi.mock('next/image', () => ({
    __esModule: true,
    default: (props: any) => <img {...props} />,
}));

// Mock asset import
vi.mock('@/assets/images/23041868.jpeg', () => ({
    default: {
        src: '/mock-image.jpg',
        height: 100,
        width: 100,
    },
}));

describe('Sidebar', () => {
    const defaultProps = {
        activeSection: 'home',
        darkMode: false,
        toggleTheme: vi.fn(),
        isCollapsed: false,
        toggleCollapse: vi.fn(),
    };

    it('renders navigation items', () => {
        render(<Sidebar {...defaultProps} />);
        expect(screen.getByText('Home')).toBeInTheDocument();
        expect(screen.getByText('About')).toBeInTheDocument();
    });

    it('renders profile info when not collapsed', () => {
        render(<Sidebar {...defaultProps} />);
        expect(screen.getByText('Scientist & Engineer')).toBeInTheDocument();
    });

    it('hides profile info when collapsed', () => {
        render(<Sidebar {...defaultProps} isCollapsed={true} />);
        expect(screen.queryByText('Scientist & Engineer')).not.toBeInTheDocument();
    });

    it('calls toggleCollapse when button is clicked', () => {
        render(<Sidebar {...defaultProps} />);
        const button = screen.getByLabelText('Collapse Sidebar');
        fireEvent.click(button);
        expect(defaultProps.toggleCollapse).toHaveBeenCalled();
    });
});
