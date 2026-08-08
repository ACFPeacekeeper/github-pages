import React from 'react';
import Link from 'next/link';

interface SidebarItemProps {
  href: string;
  icon: React.ReactNode;
  label: string;
  active: boolean;
  isCollapsed: boolean;
}

const SidebarItem: React.FC<SidebarItemProps> = ({ 
  href, 
  icon, 
  label, 
  active, 
  isCollapsed 
}) => {
  return (
    <Link 
      href={href}
      className={`flex items-center gap-3 p-3 rounded-lg transition-colors group ${
        active 
          ? 'bg-blue-50 text-blue-600 dark:bg-blue-900/20 dark:text-blue-400' 
          : 'text-slate-500 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800'
      } ${isCollapsed ? 'justify-center' : ''}`}
    >
      <div className={`transition-colors ${active ? 'text-blue-600 dark:text-blue-400' : 'group-hover:text-slate-900 dark:group-hover:text-slate-200'}`}>
        {icon}
      </div>
      
      {!isCollapsed && (
        <span className="font-medium truncate">{label}</span>
      )}
    </Link>
  );
};

export default SidebarItem;