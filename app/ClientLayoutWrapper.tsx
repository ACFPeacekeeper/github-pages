'use client';

import React, { useState, useEffect } from 'react';
import { X } from 'lucide-react';
import { usePathname } from 'next/navigation';
import Sidebar from '../src/frameworks/react/components/layout/Sidebar';
import Footer from '../src/frameworks/react/components/layout/Footer';
import Header from '../src/frameworks/react/components/layout/Header';
import { useAppDispatch } from '../src/redux/store/hooks';
import { setTheme } from '../src/redux/actions/appActions';
import { persistTheme, readStoredTheme } from '../src/redux/services/persistence';
import ReduxProvider from '../src/redux/store/ReduxProvider';

interface ClientLayoutWrapperProps {
  children: React.ReactNode;
}

const ClientLayoutContent: React.FC<ClientLayoutWrapperProps> = ({ children }) => {
  const [darkMode, setDarkMode] = useState<boolean>(true); // Default to true (Dark Mode)
  const [mobileMenuOpen, setMobileMenuOpen] = useState<boolean>(false);
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState<boolean>(true);

  const pathname = usePathname();
  const dispatch = useAppDispatch();

  // --- getActiveSection Logic ---
  const getActiveSection = (path: string | null) => {
    if (!path || path === '/') return 'home';
    const match = path.match(/^\/(?:content\/)?([a-z-]+)/);
    if (match) {
      return match[1];
    }
    return 'home';
  };

  const activeSection = getActiveSection(pathname);

  // --- Dark Mode Logic ---
  useEffect(() => {
    const storedTheme = readStoredTheme();

    // If stored as 'light', set false. Otherwise default to true (Dark).
    setDarkMode(storedTheme !== 'light');
  }, []);

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
      persistTheme('dark');
      dispatch(setTheme('dark'));
    } else {
      document.documentElement.classList.remove('dark');
      persistTheme('light');
      dispatch(setTheme('light'));
    }
  }, [darkMode, dispatch]);

  const toggleTheme = () => setDarkMode(!darkMode);
  const toggleSidebarCollapsed = () => setIsSidebarCollapsed(!isSidebarCollapsed);

  // --- Mobile Menu Logic ---
  const toggleMenu = () => setMobileMenuOpen(!mobileMenuOpen);

  useEffect(() => {
    setMobileMenuOpen(false);
  }, [pathname]);


  return (
    // FIX 1: Add 'flex flex-col' here. This stacks the Header and Content vertically.
    <div className={`flex flex-col h-screen overflow-hidden transition-colors duration-300 font-sans ${darkMode ? 'dark bg-slate-900 text-slate-200' : 'bg-slate-50 text-slate-800'}`}>

      {/* Header takes its natural height */}
      <Header darkMode={darkMode} toggleTheme={toggleTheme} />

      {/* FIX 2: Change 'h-full' to 'flex-1 w-full'. This forces the div to fill only the REMAINING space after Header. */}
      {/* Added 'overflow-hidden' here to ensure inner scrollbars work correctly relative to this container */}
      <div className="flex flex-1 w-full max-w-7xl mx-auto overflow-hidden">

        {/* Sidebar for Desktop */}
        <div className="hidden lg:block h-full">
          <Sidebar
            activeSection={activeSection}
            darkMode={darkMode}
            toggleTheme={toggleTheme}
            isCollapsed={isSidebarCollapsed}
            toggleCollapse={toggleSidebarCollapsed}
          />
        </div>

        {/* Sidebar for Mobile */}
        {mobileMenuOpen && (
          <div className="lg:hidden fixed inset-0 z-40 bg-white dark:bg-slate-900 overflow-y-auto">
            <Sidebar
              activeSection={activeSection}
              darkMode={darkMode}
              toggleTheme={toggleTheme}
              isCollapsed={false}
              toggleCollapse={toggleSidebarCollapsed}
            />

            <button
              onClick={toggleMenu}
              className="absolute top-4 right-4 p-2 rounded-full bg-slate-100 dark:bg-slate-800 text-slate-800 dark:text-slate-200 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors z-50"
              aria-label="Close menu"
            >
              <X size={24} />
            </button>
          </div>
        )}

        {/* Main Content Area */}
        {/* h-full works here now because the parent (flex-1) has a constrained height */}
        <div className="flex-1 h-full overflow-y-auto">
          <main className="flex-1 p-6 lg:p-12 w-full max-w-4xl mx-auto pt-20 lg:pt-12">
            {children}

            <Footer
              darkMode={darkMode}
              toggleTheme={toggleTheme}
            />
          </main>
        </div>
      </div>
    </div>
  );
};

const ClientLayoutWrapper: React.FC<ClientLayoutWrapperProps> = ({ children }) => (
  <ReduxProvider><ClientLayoutContent>{children}</ClientLayoutContent></ReduxProvider>
);

export default ClientLayoutWrapper;
