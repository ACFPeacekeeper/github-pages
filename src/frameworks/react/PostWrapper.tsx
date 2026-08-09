'use client';

import React, { useEffect } from 'react';
import '@/src/styles/content.css';
import '@/src/styles/posts.css';

interface PostWrapperProps {
  content: string; // The raw HTML content string (not filename)
  title?: string;
}

// ====================================================================
// STANDALONE UTILITY FUNCTION FOR SYNTAX HIGHLIGHTING
// Includes guard to prevent Maximum Call Stack Size Exceeded error.
// ====================================================================
const highlightFunctionCalls = () => {
  // --- COLOR MAPPING ---
  const PURPLE_KEYWORDS = [
    'and', 'as', 'assert', 'async', 'await', 'break', 'case', 'class', 'continue', 
    'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 
    'if', 'import', 'in', 'is', 'lambda', 'match', 'nonlocal', 'not', 'or', 
    'pass', 'raise', 'return', 'try', 'while', 'with', 'yield'
  ];
  
  const classMap = {
    keyword: 'hljs-keyword', // Purple
    literal: 'hljs-number', // Grey
    string: 'hljs-string', // Green
    blueIdentifier: 'hljs-custom-func-call', // Blue (for function names)
    orangeSymbol: 'hljs-symbol-op', // Orange (for operators and punctuation like =, +, *, (, ), ;)
  };

  // List of tokens that should be GREY
  const GREY_LITERALS_AND_TYPES = [
    'True', 'False', 'None', 'int', 'float', 'complex', 'str', 'list', 'tuple', 
    'dict', 'set', 'bool', 'frozenet', 'bytes', 'bytearray', 'memoryview', 'NoneType'
  ];

  // Regex to split code into ALL tokens
  // This tokenizer handles keywords, identifiers, numbers, strings, symbols, and whitespace.
  const TOKENIZER = /(\b[a-zA-Z_][a-zA-Z0-9_]*\b|\b\d+(\.\d+)?\b|("|').*?(\3)|[^\s\w]|\s+)/g;

  document.querySelectorAll('pre code').forEach(codeBlock => {
    let rawContent = codeBlock.textContent || '';
    
    // Temporarily decode standard entities just in case they survived the initial DOM pass
    rawContent = rawContent.replace(/&lt;/g, '<')
                            .replace(/&gt;/g, '>')
                            .replace(/&amp;/g, '&'); 

    let highlightedTokens: string[] = [];
    const localTokenizer = new RegExp(TOKENIZER, 'g');
    localTokenizer.lastIndex = 0; 
    
    let match: RegExpExecArray | null;
    let previousIndex = 0; // Guard variable to track last successful match index

    while ((match = localTokenizer.exec(rawContent)) !== null) {
      const token = match[0];
      const index = match.index;

      // Crucial Guard: If the tokenizer hasn't moved forward, prevent infinite loop.
      if (localTokenizer.lastIndex === previousIndex) {
          localTokenizer.lastIndex++; // Manually advance to skip the problematic spot
          continue;
      }
      previousIndex = localTokenizer.lastIndex;

      // Skip over pure whitespace, let it fall through to be added as a raw token
      if (token.match(/^\s+$/)) {
          highlightedTokens.push(token);
          continue;
      }

      let className: string | null = null;
      
      // 3.1. Classification Checks
      
      // --- Identifier Check ---
      const isIdentifier = /^[a-zA-Z_][a-zA-Z0-9_]*$/.test(token);

      if (isIdentifier) {
        // A. Check for Keywords (Purple)
        if (PURPLE_KEYWORDS.includes(token)) {
          className = classMap.keyword;
        }
        // B. Check for Literals and Types (Grey)
        else if (GREY_LITERALS_AND_TYPES.includes(token)) {
          className = classMap.literal;
        }
        // C. Check for Function Call Pattern (Blue)
        else {
          let lookaheadIndex = index + token.length; 
          let foundOpenParen = false;
          
          // Lookahead logic remains the same
          while (lookaheadIndex < rawContent.length) {
            const nextTokenMatch = rawContent.substring(lookaheadIndex).match(/(\s*([^\s\w]|\b[a-zA-Z_][a-zA-Z0-9_]*\b))/);
            
            if (!nextTokenMatch) break; 

            const nextToken = nextTokenMatch[2]; 

            if (!nextToken.match(/^\s+$/)) {
              if (nextToken === '(') {
                foundOpenParen = true;
              }
              break; 
            }
            lookaheadIndex += nextTokenMatch[0].length;
          }

          if (foundOpenParen) {
            className = classMap.blueIdentifier;
          }
        }
      } 
      
      // --- Non-Identifier Checks ---
      
      // D. Strings (Green)
      else if (token.match(/^("|').*?(\1)$/)) {
          className = classMap.string;
      } 
      // E. Numeric Literals (Grey)
      else if (token.match(/^\b\d+(\.\d+)?\b$/)) {
          className = classMap.literal;
      }
      // F. Symbols and Punctuation (Orange)
      else if (token.match(/^[^\s\w]$/)) { // Matches any single non-alphanumeric, non-whitespace character
          className = classMap.orangeSymbol;
      }
      
      // 3.2. Apply Span or push raw token
      if (className) {
          highlightedTokens.push(`<span class="${className}">${token}</span>`);
      } else {
          // Fallthrough: Variables, complex symbols/punctuation
          highlightedTokens.push(token);
      }
    }

    // --- 4. Update the innerHTML ---
    codeBlock.innerHTML = highlightedTokens.join('');
  });
};
// ====================================================================


const loadMathJax = () => {
  if (typeof window.MathJax === 'undefined') {
    window.MathJax = {
      tex: {
        inlineMath: [['$', '$'], ['\\(', '\\)']], 
        displayMath: [['$$', '$$'], ['\\[', '\\]']], 
      },
      svg: {
        fontCache: 'global'
      }
    };
  }
  
  if (typeof window.MathJax !== 'undefined' && typeof window.MathJax.typeset === 'function') {
    window.MathJax.typesetClear!();
    window.MathJax.typeset!();
    return;
  }

  const mathJaxScript = document.createElement('script');
  mathJaxScript.id = 'MathJax-script';
  mathJaxScript.async = true;
  mathJaxScript.src = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js";
  document.head.appendChild(mathJaxScript);
};


const PostWrapper: React.FC<PostWrapperProps> = ({ content }) => {
  
  // Consolidate both functions into one useEffect to control order
  useEffect(() => {
    if (!content) return;

    // 1. Run Syntax Highlighter
    const container = document.querySelector('.post-container');
    if (container) {
      highlightFunctionCalls(); 
    }

    // 2. Run MathJax Typesetting
    // We add a slight delay to ensure the DOM changes from highlighting are settled
    const timer = setTimeout(() => {
      loadMathJax();
    }, 50); // 50ms delay

    return () => clearTimeout(timer); // Cleanup timer on unmount/re-render
    
  }, [content]); // Reruns when content changes
  
  return (
    <div className="bg-white dark:bg-slate-900 rounded-xl shadow-lg overflow-hidden border border-slate-200 dark:border-slate-800 p-8">
      <div 
        className="post-container"
        // This is where React renders the content, which both highlighter and MathJax rely on.
        dangerouslySetInnerHTML={{ __html: content }} 
      />
    </div>
  );
};

export default PostWrapper;

declare global {
  interface Window {
    MathJax: {
      tex: {
        inlineMath: string[][];
        displayMath: string[][];
      };
      svg: {
        fontCache: string;
      };
      typesetClear?: () => void; 
      typeset?: () => void; 
    } | undefined;
  }
}