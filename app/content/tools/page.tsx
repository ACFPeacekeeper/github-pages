import React from 'react';
import { Terminal, Palette, Box, Database, Globe, Brain } from 'lucide-react';
import SectionHeading from '@/src/frameworks/react/components/ui/SectionHeading';
import GlassCard from '@/src/frameworks/react/components/ui/GlassCard';
import Badge from '@/src/frameworks/react/components/ui/Badge';
import pageImage from '@/assets/images/maxresdefault.jpg';

export default function ToolsPage() {

  const renderToolGroup = (tools: { name: string, url: string }[]) => (
    <div className="flex flex-wrap gap-2 mt-4">
      {tools.map(tool => (
        <a key={tool.name} href={tool.url} target="_blank" rel="noopener noreferrer">
          <Badge variant="outline" className="hover:bg-blue-50 dark:hover:bg-blue-900/20 cursor-pointer transition-colors">
            {tool.name}
          </Badge>
        </a>
      ))}
    </div>
  );

  return (
    <div className="space-y-10 animate-in fade-in duration-500">

      <div className="flex justify-center mb-8">
        <div className="relative w-full max-w-2xl aspect-[21/9] rounded-2xl overflow-hidden shadow-2xl border border-white/20">
          <img
            src={pageImage.src}
            alt="Jinx with Minigun"
            className="object-cover w-full h-full hover:scale-105 transition-transform duration-700"
          />
        </div>
      </div>

      <div className="text-center mb-12">
        <h1 className="text-4xl font-display font-bold text-slate-900 dark:text-white">Tools & Technologies</h1>
        <p className="text-slate-600 dark:text-slate-400 mt-2">The arsenal I use to build, design, and deploy.</p>
      </div>

      <section>
        <SectionHeading title="Programming Languages" icon={<Terminal className="text-blue-500" />} />
        <GlassCard className="p-6">
          {renderToolGroup([
            { name: 'Python', url: 'https://www.python.org/' },
            { name: 'Java', url: 'https://www.java.com/' },
            { name: 'JavaScript', url: 'https://developer.mozilla.org/en-US/docs/Web/JavaScript' },
            { name: 'TypeScript', url: 'https://www.typescriptlang.org/' },
            { name: 'C', url: 'https://devdocs.io/c/' },
            { name: 'C++', url: 'https://cplusplus.com/' },
            { name: 'Kotlin', url: 'https://kotlinlang.org/' },
            { name: 'SQL', url: 'https://www.microsoft.com/en-us/sql-server/' },
            { name: 'Bash', url: 'https://www.gnu.org/software/bash/' }
          ])}
        </GlassCard>
      </section>

      <section>
        <SectionHeading title="Frameworks & Libraries" icon={<Box className="text-red-500" />} />
        <GlassCard className="p-6">
          {renderToolGroup([
            { name: 'React', url: 'https://reactjs.org/' },
            { name: 'Next.js', url: 'https://nextjs.org/' },
            { name: 'Tailwind CSS', url: 'https://tailwindcss.com/' },
            { name: 'Bootstrap', url: 'https://getbootstrap.com/' },
            { name: 'PyTorch', url: 'https://pytorch.org/' },
            { name: 'TensorFlow', url: 'https://www.tensorflow.org/' },
            { name: 'Pandas', url: 'https://pandas.pydata.org/' },
            { name: 'NumPy', url: 'https://numpy.org/' }
          ])}
        </GlassCard>
      </section>

      <section>
        <SectionHeading title="DevOps & Build Tools" icon={<Database className="text-green-500" />} />
        <GlassCard className="p-6">
          {renderToolGroup([
            { name: 'Git', url: 'https://git-scm.com/' },
            { name: 'GitHub', url: 'https://github.com/' },
            { name: 'GitLab', url: 'https://gitlab.com/' },
            { name: 'BitBucket', url: 'https://bitbucket.org/' },
            { name: 'Docker', url: 'https://www.docker.com/' },
            { name: 'Anaconda', url: 'https://www.anaconda.com/download/' },
            { name: 'UV', url: 'https://docs.astral.sh/uv/' },
            { name: 'Maven', url: 'https://maven.apache.org/' },
            { name: 'Gradle', url: 'https://gradle.org/' },
            { name: 'PostgreSQL', url: 'https://www.postgresql.org/' },
          ])}
        </GlassCard>
      </section>

      <section>
        <SectionHeading title="Engines & Editors" icon={<Globe className="text-orange-500" />} />
        <GlassCard className="p-6">
          {renderToolGroup([
            { name: 'Node.js', url: 'https://nodejs.org/en' },
            { name: 'Unity', url: 'https://unity.com/' },
            { name: 'Neovim', url: 'https://neovim.io/' },
            { name: 'Emacs', url: 'https://www.gnu.org/software/emacs/' },
            { name: 'Visual Studio Code', url: 'https://code.visualstudio.com/' },
            { name: 'Obsidian', url: 'https://obsidian.md/' },
            { name: 'Jupyter', url: 'https://jupyter.org/' },
            { name: 'Antigravity', url: 'https://antigravity.google/' },
          ])}
        </GlassCard>
      </section>

      <section>
        <SectionHeading title="Artificial Intelligence" icon={<Brain className="text-pink-500" />} />
        <GlassCard className="p-6">
          {renderToolGroup([
            { name: 'ChatGPT', url: 'https://chat.openai.com/' },
            { name: 'Claude', url: 'https://claude.ai/' },
            { name: 'Hugging Face', url: 'https://huggingface.co/' },
            { name: 'Ollama', url: 'https://ollama.ai/' },
            { name: 'Gemini', url: 'https://gemini.ai/' },
            { name: 'Grok', url: 'https://grok.ai/' }
          ])}
        </GlassCard>
      </section>

      <section>
        <SectionHeading title="Art & Design" icon={<Palette className="text-purple-500" />} />
        <GlassCard className="p-6">
          {renderToolGroup([
            { name: 'Blender', url: 'https://www.blender.org/' },
            { name: 'LibreSprite', url: 'https://libresprite.github.io/' },
            { name: 'GNU Image Manipulation Program', url: 'https://www.gimp.org/' },
            { name: 'Figma', url: 'https://www.figma.com/' }
          ])}
        </GlassCard>
      </section>
    </div>
  );
}