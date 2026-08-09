import React from 'react';

interface GlassCardProps {
  children: React.ReactNode;
  className?: string;
}

const GlassCard: React.FC<GlassCardProps> = ({ children, className = '' }) => (
  <div className={`backdrop-blur-md bg-white/70 dark:bg-slate-800/70 border border-white/50 dark:border-white/10 shadow-sm rounded-xl transition-all duration-300 ${className}`}>
    {children}
  </div>
);

export default GlassCard;