export interface BlogPost {
  id: string;
  date: string;
  category: string;
  title: string;
  excerpt: string;
  tags: string[];
}

import type { ReactNode } from 'react';

export interface ProjectLink {
  label: string;
  url: string;
  icon: ReactNode;
}

export interface Project {
  id: string;
  title: string;
  description: string;
  icon?: ReactNode;
  iconColorClass: string;
  stats?: string;
  links: ProjectLink[];
}
