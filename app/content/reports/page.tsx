'use client';

import React from 'react';
import Link from 'next/link';
import { FileText, ExternalLink } from 'lucide-react';
import SectionHeading from '@/src/frameworks/react/components/ui/SectionHeading';
import GlassCard from '@/src/frameworks/react/components/ui/GlassCard';
import pageImage from '@/assets/images/Jinx-League-of-Legends-League-of-Legends-arcane-Netflix-TV-Series-tv-series-video-game-characters-2233556.jpg';

// Note: 'id' matches the keys in page.tsx 'REPORTS_DATA'
const REPORTS = [
  { id: 'anime-gen', title: 'Anime Generation Model', category: 'Deep Learning' },
  { id: 'gen-arch', title: 'Generative Architecture', category: 'Architecture' },
  { id: 'local-ai', title: 'Local AI Coding', category: 'DevOps' },
  { id: 'semantic', title: 'Semantic Search', category: 'NLP' },
  { id: 'strategic', title: 'Strategic Generative Pipeline', category: 'Strategy' },
  { id: 'timeseries', title: 'Time Series Forecasting', category: 'Data Science' },
  { id: 'vrp', title: 'Vehicle Routing Problem', category: 'Optimization' },
  { id: 'pcvrp', title: 'Periodic Capacitated Vehicle Routing Problem', category: 'Optimization' },
  { id: 'audio-signal-proc', title: 'Audio Signal Processing', category: 'Deep Learning' },
  { id: 'waste-logistics-architecture', title: 'Waste Logistics Architecture', category: 'Optimization' },
];

export default function ReportsIndexPage() {
  return (
    <div className="space-y-10 animate-in fade-in duration-500">
      <div className="flex justify-center mb-8">
        <div className="relative w-full max-w-2xl aspect-video rounded-2xl overflow-hidden shadow-2xl border border-white/20">
          {/* Note: Ensure this image path is correct in your public folder */}
          <img
            src={pageImage.src}
            alt="Jinx from Arcane"
            className="object-cover w-full h-full hover:scale-105 transition-transform duration-700"
          />
        </div>
      </div>

      <div className="text-center mb-12">
        <h1 className="text-4xl font-display font-bold text-slate-900 dark:text-white">Research Reports</h1>
        <p className="text-slate-600 dark:text-slate-400 mt-2">Detailed technical analysis and strategic documents.</p>
      </div>

      <section>
        <SectionHeading title="Available Reports" icon={<FileText className="text-blue-500" />} />
        <div className="grid md:grid-cols-2 gap-6">
          {REPORTS.map((report) => (
            <Link key={report.id} href={`/content/reports/${report.id}`} className="group">
              <GlassCard className="p-6 h-full hover:border-blue-500/50 transition-colors cursor-pointer">
                <div className="flex justify-between items-start">
                  <div>
                    <span className="text-xs font-bold text-blue-500 uppercase tracking-wider">{report.category}</span>
                    <h3 className="text-xl font-bold text-slate-900 dark:text-white mt-1 group-hover:text-blue-500 transition-colors">{report.title}</h3>
                  </div>
                  <ExternalLink size={18} className="text-slate-400 group-hover:text-blue-500" />
                </div>
                <p className="text-sm text-slate-500 mt-4">Click to view full HTML report.</p>
              </GlassCard>
            </Link>
          ))}
        </div>
      </section>
    </div>
  );
}