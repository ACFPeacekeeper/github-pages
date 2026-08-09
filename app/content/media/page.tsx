import React from 'react';
import { Gamepad2, Film, BookOpen, Music } from 'lucide-react';
import SectionHeading from '@/src/frameworks/react/components/ui/SectionHeading';
import GlassCard from '@/src/frameworks/react/components/ui/GlassCard';
import pageImage from '@/assets/images/dg7za3r-5c28683b-d2e4-4018-bf8b-eaea55cde631.png'; 

export default function MediaPage() {
  return (
    <div className="space-y-10 animate-in fade-in duration-500 max-w-6xl mx-auto pb-16">

        <div className="flex justify-center mb-8">
            <div className="relative w-full max-w-2xl aspect-square rounded-2xl overflow-hidden shadow-2xl border border-white/20">
                <img 
                    src={pageImage.src}
                    alt="Jinx from Arcane"
                    className="object-cover w-full h-full hover:scale-105 transition-transform duration-700"
                />
            </div>
        </div>
      
        <div className="text-center mb-12">
            <h1 className="text-4xl font-display font-bold text-slate-900 dark:text-white">Media Log</h1>
            <p className="text-slate-600 dark:text-slate-400 mt-2">Tracking what I watch, play, read, and listen to.</p>
        </div>

        <div className="grid md:grid-cols-2 gap-8">
        
            {/* Games & Print Column */}
            <div className="space-y-8">
                <section>
                    <SectionHeading title="Games" icon={<Gamepad2 className="text-purple-500" />} />
                    <GlassCard className="p-6 space-y-6">
                        <div>
                            <h3 className="font-bold text-lg mb-2 text-slate-800 dark:text-slate-200">Video Games</h3>
                            <ul className="list-disc list-inside mt-2 text-slate-600 dark:text-slate-400">
                                <li>League of Legends</li>
                                <li>Counter-Strike 2</li>
                                <li>Pokemon Series</li>
                            </ul>
                        </div>
                        <div>
                            <h3 className="font-bold text-lg mb-2 text-slate-800 dark:text-slate-200">Tabletop Games</h3>
                            <ul className="list-disc list-inside mt-2 text-slate-600 dark:text-slate-400">
                                <li>Dungeons & Dragons (5e)</li>
                                <li>Magic: The Gathering (Commander)</li>
                                <li>Catan</li>
                                <li>Monopoly</li>
                            </ul>
                        </div>
                    </GlassCard>
                </section>

                <section>
                    <SectionHeading title="Print Media" icon={<BookOpen className="text-blue-500" />} />
                    <GlassCard className="p-6">
                        <h3 className="font-bold text-lg mb-2 text-slate-800 dark:text-slate-200">Papers</h3>
                        <ul className="list-disc list-inside mt-2 text-slate-600 dark:text-slate-400">
                            <li>Attention Is All You Need</li>
                        </ul>
                        <h3 className="font-bold text-lg mb-2 text-slate-800 dark:text-slate-200">Books</h3>
                        <ul className="list-disc list-inside mt-2 text-slate-600 dark:text-slate-400">
                            <li>Reinforcement Learning: An Introduction</li>
                        </ul>
                    </GlassCard>
                </section>
            </div>

            {/* Cinema & Music Column */}
            <div className="space-y-8">
                <section>
                    <SectionHeading title="Cinema & TV" icon={<Film className="text-red-500" />} />
                    <GlassCard className="p-6 space-y-6">
                        <div>
                            <h3 className="font-bold text-lg mb-2 text-slate-800 dark:text-slate-200">Animated Movies</h3>
                            <ul className="list-disc list-inside mt-2 text-slate-600 dark:text-slate-400">
                                <li>Carmen Sandiego: To Steal or Not to Steal</li>
                                <li>Next Gen</li>
                            </ul>
                        </div>
                        
                        <div>
                            <h3 className="font-bold text-lg mb-2 text-slate-800 dark:text-slate-200">Animated Mini-Series & OVAs</h3>
                            <ul className="list-disc list-inside mt-2 text-slate-600 dark:text-slate-400">
                                <li>Justice League x RWBY: Super Heroes and Huntsmen</li>
                                <li>Pokemon: The Arceus Chronicles</li>
                                <li>Star Wars: Tales of the Empire</li>
                                <li>Star Wars: Tales of the Jedi</li>
                            </ul>
                        </div>

                        <div>
                            <h3 className="font-bold text-lg mb-2 text-slate-800 dark:text-slate-200">Animated Series</h3>
                            <ul className="list-disc list-inside mt-2 text-slate-600 dark:text-slate-400">
                                <li>Arcane</li>
                                <li>Carmen Sandiego S4</li>
                                <li>Ergo Proxy</li>
                                <li>Glitch Techs S2</li>
                                <li>Harley Quinn S4</li>
                                <li>Kimi to Boku no Saigo no Senjou, Aruiwa Sekai ga Hajimaru Seisen</li>
                                <li>Little Demon</li>
                                <li>Sherwood</li>
                                <li>Sonic Prime S3</li>
                                <li>Star Trek: Lower Decks S4</li>
                                <li>Star Trek: Prodigy S2</li>
                                <li>Star Wars: The Bad Batch S3</li>
                                <li>Teen Titans S1</li>
                                <li>The Dragon Prince S6</li>
                                <li>The Legend of Vox Machina S2</li>
                                <li>Velma S2</li>
                            </ul>
                        </div>
                        
                        <div>
                            <h3 className="font-bold text-lg mb-2 text-slate-800 dark:text-slate-200">Live Action</h3>
                            <ul className="list-disc list-inside mt-2 text-slate-600 dark:text-slate-400">
                                <li>Star Wars: The Acolyte</li>
                                <li>Knuckles</li>
                                <li>The Mandalorian</li>
                            </ul>
                        </div>
                    </GlassCard>
                </section>

                <section>
                    <SectionHeading title="Music" icon={<Music className="text-yellow-500" />} />
                    <GlassCard className="p-6 space-y-6">
                         <div>
                            <h3 className="font-bold text-lg mb-2 text-slate-800 dark:text-slate-200">Music Bands</h3>
                            <ul className="list-disc list-inside mt-2 text-slate-600 dark:text-slate-400">
                                <li>Linkin Park </li>
                                <li>Imagine Dragons</li>
                            </ul>
                            <h3 className="font-bold text-lg mb-2 text-slate-800 dark:text-slate-200">Original Soundtracks</h3>
                            <ul className="list-disc list-inside mt-2 text-slate-600 dark:text-slate-400">
                                <li>Arcane</li>
                            </ul>
                        </div>
                    </GlassCard>
                </section>
            </div>

        </div>
    </div>
  );
}