import React from 'react';
import { Moon, Sun } from 'lucide-react';

interface FooterProps {
  darkMode: boolean;
  toggleTheme: () => void;
}

const Footer: React.FC<FooterProps> = ({ darkMode, toggleTheme }) => (
    <footer className="border-t border-slate-200 dark:border-slate-800 pt-8 mt-20 pb-8 text-center md:text-left text-slate-500 text-sm">
        <div className="flex flex-col md:flex-row justify-between items-center">
            <p>&copy; {new Date().getFullYear()} ACFHarbinger. All rights reserved.</p>

            <div className="mt-4 md:mt-0 flex items-center gap-6">
                <a href="#" className="hover:text-slate-900 dark:hover:text-slate-200 transition-colors">RSS</a>
                <a href="#" className="hover:text-slate-900 dark:hover:text-slate-200 transition-colors">Privacy</a>
                <a href="#" className="hover:text-slate-900 dark:hover:text-slate-200 transition-colors">Sitemap</a>

                {/* Theme Toggle Button */}
                <button
                    onClick={toggleTheme}
                    className="hover:text-yellow-500 transition-colors flex items-center justify-center"
                    aria-label="Toggle Dark Mode"
                >
                    {darkMode ? <Sun size={18} /> : <Moon size={18} />}
                </button>
            </div>
          </div>
    </footer>
);

export default Footer;
