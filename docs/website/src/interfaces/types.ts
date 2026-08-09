export interface BlogPost {
  id: string;
  date: string;
  category: string;
  title: string;
  excerpt: string;
  tags: string[];
}

export interface LoreStory {
  id: string;
  title: string;
  era: string;
  summary: string;
  tags: string[];
  docPath?: string;
}
