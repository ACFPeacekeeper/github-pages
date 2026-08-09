import React, { ReactNode } from 'react';

// Define the expected props interface
interface SectionHeadingProps {
  title: string;
  icon: ReactNode;
  className?: string;
}

const SectionHeading: React.FC<SectionHeadingProps> = ({ title, icon, className = '' }) => {
  return (
    // Standard section heading structure using the passed props
    <div className={`flex items-center space-x-3 mb-4 ${className}`}>
      {/* Container for the icon */}
      <div className="p-2 rounded-lg bg-slate-100 dark:bg-slate-800 shadow-md flex items-center justify-center">
        {icon}
      </div>
      {/* The title text */}
      <h2 className="text-2xl font-bold font-display text-slate-900 dark:text-white">
        {title}
      </h2>
    </div>
  );
};

export default SectionHeading;