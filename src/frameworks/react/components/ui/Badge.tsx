import React, { ReactNode } from 'react';

// Define possible variants for the badge
type BadgeVariant = 'default' | 'outline';

interface BadgeProps {
  children: ReactNode;
  /** Custom class names to apply to the badge. */
  className?: string;
  /** Defines the visual style of the badge (default or outline). */
  variant?: BadgeVariant;
}

const Badge: React.FC<BadgeProps> = ({ children, className = '', variant = 'default' }) => {
  let baseClasses = 'inline-flex items-center rounded-full px-3 py-1 text-xs font-medium transition-colors';
  
  // Define default styles based on variant
  if (variant === 'outline') {
    // Styles for the 'outline' variant
    baseClasses += ' border border-slate-300 dark:border-slate-700 text-slate-700 dark:text-slate-300 bg-transparent';
  } else {
    // Styles for the 'default' variant (e.g., solid blue)
    baseClasses += ' bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300';
  }

  // Combine base classes with any user-provided classes
  const finalClassName = `${baseClasses} ${className}`;

  return (
    <span className={finalClassName}>
      {children}
    </span>
  );
};

export default Badge;