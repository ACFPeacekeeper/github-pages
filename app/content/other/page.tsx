'use client';

import React from 'react';
import Link from 'next/link';
import { FileText, ExternalLink } from 'lucide-react';
import SectionHeading from '@/src/frameworks/react/components/ui/SectionHeading';
import GlassCard from '@/src/frameworks/react/components/ui/GlassCard';
import pageImage from '@/assets/images/GcxP4GkXMAAX7az.jpeg';

const OTHER_ITEMS = [
    { id: 'math-curriculum', title: 'Mathematics Curriculum', category: 'Education' },
];

export default function OtherIndexPage() {
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
                <h1 className="text-4xl font-display font-bold text-slate-900 dark:text-white">Other Stuff</h1>
                <p className="text-slate-600 dark:text-slate-400 mt-2">Miscellaneous ramblings and roadmaps.</p>
            </div>

            <section>
                <SectionHeading title="Available Stuff" icon={<FileText className="text-blue-500" />} />
                <div className="grid md:grid-cols-2 gap-6">
                    {OTHER_ITEMS.map((item) => (
                        <Link key={item.id} href={`/content/other/${item.id}`} className="group">
                            <GlassCard className="p-6 h-full hover:border-blue-500/50 transition-colors cursor-pointer">
                                <div className="flex justify-between items-start">
                                    <div>
                                        <span className="text-xs font-bold text-blue-500 uppercase tracking-wider">{item.category}</span>
                                        <h3 className="text-xl font-bold text-slate-900 dark:text-white mt-1 group-hover:text-blue-500 transition-colors">{item.title}</h3>
                                    </div>
                                    <ExternalLink size={18} className="text-slate-400 group-hover:text-blue-500" />
                                </div>
                                <p className="text-sm text-slate-500 mt-4">Click to view project.</p>
                            </GlassCard>
                        </Link>
                    ))}
                </div>
            </section>
        </div>
    );
}
