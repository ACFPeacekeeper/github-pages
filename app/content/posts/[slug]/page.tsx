import React from 'react';
import { notFound } from 'next/navigation';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import fs from 'fs';
import path from 'path';
import PostWrapper from '@/src/frameworks/react/PostWrapper';

// Define the type for the URL parameters
interface PostPageProps {
  params: {
    slug: string;
  };
}

// Map slugs to the corresponding HTML file names
const POSTS_HTML_MAP: Record<string, string> = {
  'Attention_Learn_to_Solve_Routing_Problem': 'Attention_Learn_to_Solve_Routing_Problem.html',
  'Combinatorial_Optimization_an_Introduction': 'Combinatorial_Optimization_an_Introduction.html',
  'Notes_on_RL_an_Introduction': 'Notes_on_RL_an_Introduction.html',
};

export async function generateStaticParams() {
  return Object.keys(POSTS_HTML_MAP).map((slug) => ({
    slug,
  }));
}

export default function PostPage({ params }: PostPageProps) {
  const { slug } = params;
  const htmlFileName = POSTS_HTML_MAP[slug];

  if (!htmlFileName) {
    notFound();
  }

  const filePath = path.join(process.cwd(), 'public', 'posts', htmlFileName);

  let content = '';
  try {
    const fileContent = fs.readFileSync(filePath, 'utf8');
    // Extract only the content inside the <body> tags
    const bodyMatch = fileContent.match(/<body[^>]*>([\s\S]*)<\/body>/i);
    
    // IMPORTANT: Remove the original MathJax script tag if it exists, as we are loading it manually in PostWrapper
    content = (bodyMatch ? bodyMatch[1] : fileContent)
        .replace(/<script src="https:\/\/polyfill.io\/v3\/polyfill.min.js\?features=es6"><\/script>/i, '')
        .replace(/<script id="MathJax-script" async src="https:\/\/cdn.jsdelivr.net\/npm\/mathjax@3\/es5\/tex-mml-chtml.js"><\/script>/i, '');
    
  } catch (error) {
    console.error(`Error reading file ${htmlFileName}:`, error);
    notFound();
  }

  return (
    // This is the parent container. It sets the max width for the post content.
    <div className="max-w-4xl mx-auto py-12 px-4"> 
      <div className="mb-6">
        <Link
          href="/content/posts"
          className="flex items-center text-slate-600 dark:text-slate-400 hover:text-blue-500 transition-colors"
        >
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back to Posts
        </Link>
      </div>

      {/* PostWrapper now handles the content display within this max-w-4xl boundary */}
      <PostWrapper content={content} title={slug} />
    </div>
  );
}