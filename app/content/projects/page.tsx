import React from 'react';
import Link from 'next/link';
import { Brain, Code, Cpu, Gamepad2, FileText, BookOpen } from 'lucide-react';
import SectionHeading from '@/src/components/ui/SectionHeading';
import GlassCard from '@/src/components/ui/GlassCard';
import Badge from '@/src/components/ui/Badge';
import pageImage from '@/assets/images/Ella-Purnell-Jinx-Arcane-League-of-Legends.webp';

// Common style for links
const linkStyles = "inline-flex items-center text-sm font-medium text-purple-400 hover:text-purple-300 transition-colors";

export default function ProjectsPage() {
  return (
    <div className="space-y-10 animate-in fade-in duration-500 max-w-4xl mx-auto pb-16">
      
       {/* Hero Image */}
       <div className="flex justify-center mb-8">
        <div className="relative w-full aspect-[21/9] rounded-2xl overflow-hidden shadow-2xl border border-white/20 dark:border-slate-700/50">
          <img 
            src={pageImage.src}
            alt="Jinx Arcane"
            className="object-cover w-full h-full hover:scale-105 transition-transform duration-700"
          />
        </div>
      </div>

      <div className="text-center mb-12">
        <h1 className="text-4xl font-display font-bold text-slate-900 dark:text-white">Projects</h1>
        <p className="text-slate-600 dark:text-slate-400 mt-2">A showcase of my research, development, and engineering work.</p>
      </div>

      {/* AI Research */}
      <section>
        <SectionHeading title="AI Research" icon={<Brain className="text-pink-500" />} />
        <div className="grid md:grid-cols-2 gap-6">
          <GlassCard className="p-6">
            <h3 className="text-xl font-bold text-slate-900 dark:text-white mb-2">Contrastive Learning Transformer (CLT)</h3>
            <p className="text-slate-600 dark:text-slate-300 mb-4">
              For Label Efficient Semantic Segmentation. Explored transformer architectures to improve segmentation with limited labeled data.
            </p>
            <div className="flex flex-wrap gap-2">
                <Badge>Deep Learning</Badge>
                <Badge>Computer Vision</Badge>
            </div>
          </GlassCard>

          <GlassCard className="p-6">
            <h3 className="text-xl font-bold text-slate-900 dark:text-white mb-2">Multimodal Graph Neural Network (MGNN)</h3>
            <p className="text-slate-600 dark:text-slate-300 mb-4">
              For Automatic Generation of Coherent Image Descriptions. Leveraging graph structures to connect visual features with semantic text generation.
            </p>
            <div className="flex flex-wrap gap-2">
                <Badge>GNN</Badge>
                <Badge>NLP</Badge>
            </div>
          </GlassCard>
          
          <GlassCard className="p-6">
            <h3 className="text-xl font-bold text-slate-900 dark:text-white mb-2">CSE Master of Science (MSc) Dissertation</h3>
            <p className="text-slate-600 dark:text-slate-300 mb-4">
              Thesis: &ldquo;Leveraging Deep Unsupervised Models Towards Learning Robust Multimodal Representations&rdquo;. Developed and compared new Multimodal Deep Unsupervised Models.
            </p>
            <div className="flex flex-wrap gap-4 items-center">
                <div className="flex gap-2">
                    <Badge>Unsupervised Learning</Badge>
                    <Badge>Multimodal</Badge>
                </div>
                <div className="flex gap-4">
                    <Link 
                        href="../docs/IST_UL___MEIC_Thesis___Dissertacao_final__Copy_.pdf"
                        target="_blank" 
                        rel="noopener noreferrer" 
                        className={linkStyles}
                    >
                        <BookOpen className="w-4 h-4 mr-1" /> Dissertation PDF
                    </Link>
                    <Link 
                        href="https://github.com/ACFHarbinger/rgmc" 
                        target="_blank" 
                        rel="noopener noreferrer" 
                        className={linkStyles}
                    >
                        <Code className="w-4 h-4 mr-1" /> GitHub Repository
                    </Link>
                </div>
            </div>
          </GlassCard>

           <GlassCard className="p-6">
            <h3 className="text-xl font-bold text-slate-900 dark:text-white mb-2">WSmart Route+</h3>
            <p className="text-slate-600 dark:text-slate-300 mb-4">
              Reinforcement Learning agent for Waste Collection routing optimization, focusing on the Periodic Capacitated Vehicle Routing Problem.
            </p>
            <div className="flex flex-wrap gap-4 items-center">
                <div className="flex gap-2">
                    <Badge>Reinforcement Learning</Badge>
                    <Badge>Optimization</Badge>
                </div>
                <Link 
                    href="../images/workshop_posters/workshop-poster.png"
                    target="_blank" 
                    rel="noopener noreferrer" 
                    className={linkStyles}
                >
                    <FileText className="w-4 h-4 mr-1" /> Workshop Poster
                </Link>
            </div>
          </GlassCard>
        </div>
      </section>

      {/* Web Development */}
      <section>
        <SectionHeading title="Web Development" icon={<Code className="text-blue-500" />} />
        <div className="grid md:grid-cols-2 gap-6">
          <GlassCard className="p-6">
            <h3 className="text-lg font-bold text-slate-900 dark:text-white mb-1">EBEC 2020 Website</h3>
            <p className="text-sm text-slate-500 dark:text-slate-400 mb-3">Event website for BEST Engineering Competition, themed after The Mandalorian TV series.</p>
            <Link 
                href="https://github.com/ACFHarbinger/ebec2020" 
                target="_blank" 
                rel="noopener noreferrer" 
                className={linkStyles}
            >
                <Code className="w-4 h-4 mr-1" /> GitHub Repository
            </Link>
          </GlassCard>
           <GlassCard className="p-6">
            <h3 className="text-lg font-bold text-slate-900 dark:text-white mb-1">Personal Website</h3>
            <p className="text-sm text-slate-500 dark:text-slate-400 mb-3">The website you are looking at right now, showcasing my blog posts and projects. Built with Next.js, React, and Tailwind CSS for modern design.</p>
            <Link 
                href="https://github.com/ACFHarbinger/github-pages" 
                target="_blank" 
                rel="noopener noreferrer" 
                className={linkStyles}
            >
                <Code className="w-4 h-4 mr-1" /> GitHub Repository
            </Link>
          </GlassCard>
        </div>
      </section>

      {/* Hardware & Software */}
      <section>
        <SectionHeading title="Hardware & Software" icon={<Cpu className="text-green-500" />} />
        <div className="grid md:grid-cols-2 gap-6">
          <GlassCard className="p-6">
            <h3 className="text-lg font-bold text-slate-900 dark:text-white mb-1">Hacker Entertainment System (HES)</h3>
            <p className="text-sm text-slate-500 dark:text-slate-400 mb-3">A custom game controller inspired by the classic NES, built during a FabLab workshop.</p>
            <Link 
                href="https://github.com/ACFHarbinger/HES" 
                target="_blank" 
                rel="noopener noreferrer" 
                className={linkStyles}
            >
                <Code className="w-4 h-4 mr-1" /> GitHub Repository
            </Link>
          </GlassCard>
           <GlassCard className="p-6">
            <h3 className="text-lg font-bold text-slate-900 dark:text-white mb-1">HackerSchool Robot (HS-Robot)</h3>
            <p className="text-sm text-slate-500 dark:text-slate-400 mb-3">An incomplete robotics project for autonomous navigation and item retrieval.</p>
            <Link 
                href="https://github.com/ACFHarbinger/HS-Robot-Controller" 
                target="_blank" 
                rel="noopener noreferrer" 
                className={linkStyles}
            >
                <Code className="w-4 h-4 mr-1" /> GitHub Repository
            </Link>
          </GlassCard>
        </div>
      </section>

      {/* Game Development */}
      <section>
        <SectionHeading title="Game Development" icon={<Gamepad2 className="text-purple-500" />} />
        <div className="grid md:grid-cols-2 gap-6">
          <GlassCard className="p-6">
            <h3 className="text-lg font-bold text-slate-900 dark:text-white">Project Emerald</h3>
            <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">A game development exploration.</p>
          </GlassCard>
           <GlassCard className="p-6">
            <h3 className="text-lg font-bold text-slate-900 dark:text-white">Project Fishes</h3>
            <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">Aquatic simulation and gameplay mechanics.</p>
          </GlassCard>
        </div>
      </section>
      
       {/* Bibliography */}
      <section>
        <SectionHeading title="Bibliography" icon={<FileText className="text-slate-500" />} />
        <GlassCard className="p-6">
            <h3 className="text-lg font-bold text-slate-900 dark:text-white mb-3">WSmart Route+ References</h3>
            <ul className="space-y-3 text-sm text-slate-600 dark:text-slate-400 list-disc list-inside">
                <li>
                    <strong className="text-slate-700 dark:text-slate-300">An Introduction to Deep Reinforcement Learning</strong> - François-Lavet, V., et al. (2018). 
                    <Link href="http://doi.org/10.1561/2200000071" target="_blank" rel="noopener noreferrer" className="ml-2 inline-flex items-center text-sm font-medium text-purple-400 hover:text-purple-300 transition-colors">
                        DOI
                    </Link>
                </li>
                 <li>
                    <strong className="text-slate-700 dark:text-slate-300">Reinforcement Learning for Combinatorial Optimization: A Survey</strong> - Mazyavkina, N., et al. (2020). 
                    <Link href="http://arxiv.org/abs/2003.03600" target="_blank" rel="noopener noreferrer" className="ml-2 inline-flex items-center text-sm font-medium text-purple-400 hover:text-purple-300 transition-colors">
                        arXiv
                    </Link>
                </li>
                <li>
                    <strong className="text-slate-700 dark:text-slate-300">Solving capacitated vehicle routing problem with demands as fuzzy random variable</strong> - Singh, V. P., et al. (2023). Soft Comput. 
                    <Link href="http://doi.org/10.1007/s00500-023-08888-1" target="_blank" rel="noopener noreferrer" className="ml-2 inline-flex items-center text-sm font-medium text-purple-400 hover:text-purple-300 transition-colors">
                        DOI
                    </Link>
                </li>
                 <li>
                    <strong className="text-slate-700 dark:text-slate-300">Deep Reinforcement Learning for Solving the Heterogeneous Capacitated Vehicle Routing Problem</strong> - Li, J., et al. (2022). IEEE Transactions on Cybernetics. 
                    <Link href="http://doi.org/10.1109/TCYB.2021.3111082" target="_blank" rel="noopener noreferrer" className="ml-2 inline-flex items-center text-sm font-medium text-purple-400 hover:text-purple-300 transition-colors">
                        DOI
                    </Link>
                </li>
                 <li>
                    <strong className="text-slate-700 dark:text-slate-300">The orienteering problem</strong> - Golden, B., et al. (1987). Nav Res Logist. 
                    <Link href="http://doi.org/10.1002/1520-6750(198706)34:3<307::AID-NAV3220340302>3.0.CO;2-D" target="_blank" rel="noopener noreferrer" className="ml-2 inline-flex items-center text-sm font-medium text-purple-400 hover:text-purple-300 transition-colors">
                        DOI
                    </Link>
                </li>
            </ul>
             <h3 className="text-lg font-bold text-slate-900 dark:text-white mt-6 mb-3">CSE MSc Dissertation References</h3>
            <ul className="space-y-3 text-sm text-slate-600 dark:text-slate-400 list-disc list-inside">
                 <li>
                    <strong className="text-slate-700 dark:text-slate-300">Autoencoders</strong> - Bank, D., et al. (2021). 
                    <Link href="http://arxiv.org/abs/2003.05991" target="_blank" rel="noopener noreferrer" className="ml-2 inline-flex items-center text-sm font-medium text-purple-400 hover:text-purple-300 transition-colors">
                        arXiv
                    </Link>
                </li>
                 <li>
                    <strong className="text-slate-700 dark:text-slate-300">Geometric Multimodal Contrastive Representation Learning</strong> - Poklukar, P., et al. (2022).
                    <Link href="http://arxiv.org/abs/2202.03390" target="_blank" rel="noopener noreferrer" className="ml-2 inline-flex items-center text-sm font-medium text-purple-400 hover:text-purple-300 transition-colors">
                        arXiv
                    </Link>
                </li>
                <li>
                    <strong className="text-slate-700 dark:text-slate-300">Self-Supervised Video Representation Learning With Odd-One-Out Networks</strong> - Fernando, B., et al. (2017).
                    <Link href="http://arxiv.org/abs/1611.06646" target="_blank" rel="noopener noreferrer" className="ml-2 inline-flex items-center text-sm font-medium text-purple-400 hover:text-purple-300 transition-colors">
                        arXiv
                    </Link>
                </li>
                <li>
                    <strong className="text-slate-700 dark:text-slate-300">Multimodal Generative Models for Scalable Weakly-Supervised Learning</strong> - Wu, M., & Goodman, N. (2018).
                    <Link href="http://arxiv.org/abs/1802.05335" target="_blank" rel="noopener noreferrer" className="ml-2 inline-flex items-center text-sm font-medium text-purple-400 hover:text-purple-300 transition-colors">
                        arXiv
                    </Link>
                </li>
            </ul>
        </GlassCard>
      </section>

    </div>
  );
}
