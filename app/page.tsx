import React from 'react';
import { PenTool, Calendar, Star, FileText, BookOpen, Code, ArrowRight } from 'lucide-react';
import GlassCard from '../src/frameworks/react/components/ui/GlassCard';
import SectionHeading from '../src/frameworks/react/components/ui/SectionHeading';
import Badge from '../src/frameworks/react/components/ui/Badge';
import { BlogPost, Project } from '../src/interfaces/types';
import HeroModel from '../src/frameworks/react/components/models/HeroModel';
import ResearchConstellation from '../src/frameworks/react/components/graph/ResearchConstellation';
import ConvergenceSimulation from '../src/frameworks/react/components/routes/ConvergenceSimulation';
import AudioSpectrum from '../src/frameworks/react/components/audio/AudioSpectrum';
import ResearchShelf from '../src/frameworks/react/components/books/ResearchShelf';
import FleetRouteCanvas from '../src/frameworks/react/components/canvas/FleetRouteCanvas';
import PrototypeCard from '../src/frameworks/react/components/games/PrototypeCard';
import MediaMosaic from '../src/frameworks/react/components/image/MediaMosaic';
import FleetRouteMap from '../src/frameworks/react/components/maps/FleetRouteMap';
import MediaReel from '../src/frameworks/react/components/video/MediaReel';
import { StarfieldWrapper } from '../src/frameworks/astro/components/StarfieldWrapper';
import { AureliaWrapper } from '../src/frameworks/aurelia/components/AureliaWrapper';

/**
 * Mock Data
 */
const BLOG_POSTS: BlogPost[] = [
  {
    id: 'Notes_on_RL_an_Introduction',
    date: 'Oct 31, 2024',
    category: 'Reinforcement Learning',
    title: 'Notes on RL: An Introduction',
    excerpt: 'A deep dive into the foundational concepts of Reinforcement Learning, exploring Markov Decision Processes (MDPs) and basic policy iteration methods.',
    tags: ['Basics', 'Theory']
  },
  {
    id: 'Attention_Learn_to_Solve_Routing_Problem',
    date: 'Oct 28, 2024',
    category: 'Deep Learning',
    title: 'Attention: Learn To Solve Routing Problems',
    excerpt: 'Analyzing the application of Attention mechanisms in Neural Combinatorial Optimization. How transformers can replace heuristics for TSP and VRP.',
    tags: ['Routing', 'Attention']
  },
  {
    id: 'Combinatorial_Optimization_an_Introduction',
    date: 'Oct 28, 2024',
    category: 'Math',
    title: 'Combinatorial Optimization Intro',
    excerpt: 'An introduction to the field of Combinatorial Optimization, focusing on complexity classes (P vs NP) and exact vs. heuristic solving methods.',
    tags: ['Optimization']
  }
];

// Reconstructed based on your projects.md
const PROJECTS: Project[] = [
  {
    id: 'p1',
    title: 'WSmart Route+',
    description: 'A Reinforcement Learning agent designed to optimize Waste Collection routing. This project explores how RL agents can learn efficient paths in dynamic environments compared to traditional heuristics.',
    links: [{ label: 'Workshop Poster', url: '/github-pages/images/workshop_posters/workshop-poster.png', icon: <FileText size={14} className="w-4 h-4 mr-1" /> }],
    stats: 'RL Agent',
    icon: undefined,
    iconColorClass: ''
  },
  {
    id: 'p2',
    title: 'CSE Master of Science (MSc) Dissertation',
    description: 'Thesis: "Leveraging Deep Unsupervised Models Towards Learning Robust Multimodal Representations". Developed and compared new Multimodal Deep Unsupervised Models.',
    links: [
      { label: 'Dissertation PDF', url: '/github-pages/docs/IST_UL___MEIC_Thesis___Dissertacao_final__Copy_.pdf', icon: <BookOpen size={14} className="w-4 h-4 mr-1" /> },
      { label: 'GitHub Repository', url: 'https://github.com/ACFHarbinger/rgmc', icon: <Code size={14} className="w-4 h-4 mr-1" /> }
    ],
    stats: 'GNN',
    icon: undefined,
    iconColorClass: ''
  },
  {
    id: 'p3',
    title: 'Personal Website',
    description: 'The website you are looking at right now, showcasing my blog posts and projects. Built with Next.js, React, and Tailwind CSS for modern design.',
    links: [{ label: 'GitHub Repository', url: 'https://github.com/acfharbinger', icon: <Code size={14} className="w-4 h-4 mr-1" /> }],
    stats: 'Next.js',
    icon: undefined,
    iconColorClass: ''
  }
];

export default function Home() {
  return (
    <>
      {/* Hero Section */}
      <section className="hero-observatory mb-20" aria-labelledby="hero-title">
        <div className="hero-observatory__copy">
          <Badge variant="default" className="bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300">
            <span className="status-pulse" aria-hidden="true" /> Available for research collaboration
          </Badge>
          <p className="hero-kicker">Scientist · Engineer · Educator</p>
          <h1 id="hero-title" className="hero-title">
            Learning systems for <span>hard decisions.</span>
          </h1>
          <p className="hero-summary">
            I combine <strong>deep reinforcement learning</strong> with <strong>operations research</strong> to solve combinatorial optimization problems—and turn the results into systems people can explore.
          </p>
          <div className="hero-actions">
            <a href="/github-pages/content/projects" className="hero-action hero-action--primary">
              Explore my work <ArrowRight size={18} />
            </a>
            <a href="/github-pages/content/about" className="hero-action hero-action--secondary">
              About me
            </a>
          </div>
          <dl className="hero-metrics" aria-label="Professional highlights">
            <div><dt>Focus</dt><dd>AI × OR</dd></div>
            <div><dt>Research</dt><dd>INESC-ID</dd></div>
            <div><dt>Teaching</dt><dd>IST</dd></div>
          </dl>
        </div>
        <HeroModel />
      </section>

      <ResearchConstellation />

      <ConvergenceSimulation />

      <section className="mb-20" aria-labelledby="field-notes-title">
        <div className="mb-8"><p className="hero-kicker">Field notes</p><h2 id="field-notes-title" className="mt-2 text-3xl font-black text-slate-900 dark:text-white">Research, play, and the things that keep me curious.</h2></div>
        <div className="grid gap-6 lg:grid-cols-2">
          <FleetRouteMap />
          <FleetRouteCanvas />
          <AudioSpectrum />
          <PrototypeCard />
          <ResearchShelf />
          <MediaMosaic />
          <div className="lg:col-span-2">
            <StarfieldWrapper />
          </div>
          <div className="lg:col-span-2">
            <AureliaWrapper />
          </div>
          <div className="lg:col-span-2"><MediaReel /></div>
        </div>
      </section>

      {/* Latest Posts Section */}
      <section className="mb-16">
        <div className="flex items-center justify-between mb-8">
          <SectionHeading title="Latest Posts" icon={<PenTool className="text-purple-500" />} />
          <a href="/github-pages/content/posts" className="text-sm font-medium text-blue-600 dark:text-blue-400 hover:underline flex items-center gap-1">
            View all <ArrowRight size={14} />
          </a>
        </div>

        <div className="grid gap-6">
          {BLOG_POSTS.map((post) => (
            <GlassCard key={post.id} className="group hover:border-blue-500/30 transition-colors">
              <div className="p-6">
                <div className="flex items-center gap-3 text-sm text-slate-500 mb-3">
                  <span className="flex items-center gap-1"><Calendar size={14} /> {post.date}</span>
                  <span className="w-1 h-1 rounded-full bg-slate-300 dark:bg-slate-600"></span>
                  <span className="text-blue-600 dark:text-blue-400 font-medium">{post.category}</span>
                </div>
                <h3 className="text-xl font-bold font-display text-slate-900 dark:text-white mb-2 group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors">
                  <a href={`/github-pages/content/posts/${post.id}`}>{post.title}</a>
                </h3>
                <p className="text-slate-600 dark:text-slate-400 mb-4 line-clamp-2">
                  {post.excerpt}
                </p>
                <div className="flex flex-wrap gap-2 items-center justify-between mt-auto">
                  <div className="flex gap-2">
                    {post.tags.map(tag => (
                      <Badge key={tag} variant="outline" className="text-xs">{tag}</Badge>
                    ))}
                  </div>
                  <a href={`/github-pages/content/posts/${post.id}`} className="text-sm font-medium text-blue-600 dark:text-blue-400 flex items-center gap-1 group-hover:translate-x-1 transition-transform">
                    Read more <ArrowRight size={14} />
                  </a>
                </div>
              </div>
            </GlassCard>
          ))}
        </div>
      </section>

      {/* Latest Projects Section */}
      <section>
        <div className="flex items-center justify-between mb-8">
          <SectionHeading title="Latest Projects" icon={<Code className="text-blue-500" />} />
          <a href="/github-pages/content/projects" className="text-sm font-medium text-blue-600 dark:text-blue-400 hover:underline flex items-center gap-1">
            View portfolio <ArrowRight size={14} />
          </a>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          {PROJECTS.map((project) => (
            <GlassCard key={project.id} className="h-full flex flex-col">
              <div className="p-6 flex-1">
                <h4 className="text-lg font-bold font-display flex items-center gap-3 text-slate-900 dark:text-white">
                  {project.title}
                  {project.stats && (
                    <span className="hidden sm:flex items-center gap-1 text-xs font-medium px-2 py-0.5 bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-500 rounded-full border border-yellow-200 dark:border-yellow-800">
                      <Star size={10} fill="currentColor" /> {project.stats}
                    </span>
                  )}
                </h4>
                <p className="text-slate-600 dark:text-slate-400 text-sm mt-2 mb-4 leading-relaxed">
                  {project.description}
                </p>
                <div className="flex items-center gap-4 text-sm mt-auto">
                  {project.links && project.links.map((link, idx) => (
                    <a
                      key={idx}
                      href={link.url}
                      className="inline-flex items-center text-sm font-medium text-purple-400 hover:text-purple-300 transition-colors"
                    >
                      {link.icon} {link.label}
                    </a>
                  ))}
                </div>
              </div>
            </GlassCard>
          ))}
        </div>
      </section>
    </>
  );
}
