const media = [
  { src: '/github-pages/images/Ella-Purnell-Jinx-Arcane-League-of-Legends.webp', alt: 'Jinx from Arcane', label: 'Animation' },
  { src: '/github-pages/images/maxresdefault.jpg', alt: 'A cinematic still', label: 'Film' },
  { src: '/github-pages/images/knowledge_graph.png', alt: 'A knowledge graph visualization', label: 'Visual thinking' },
];

export default function MediaMosaic() {
  return <section aria-labelledby="media-mosaic-title"><h2 id="media-mosaic-title" className="mb-4 font-bold text-slate-900 dark:text-white">Visual references</h2><div className="grid grid-cols-3 gap-2">{media.map((item) => <figure key={item.src} className="group relative aspect-square overflow-hidden rounded-xl bg-slate-200 dark:bg-slate-800"><img src={item.src} alt={item.alt} className="h-full w-full object-cover transition duration-500 group-hover:scale-110" /><figcaption className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/80 p-2 text-[.65rem] font-bold text-white">{item.label}</figcaption></figure>)}</div></section>;
}
