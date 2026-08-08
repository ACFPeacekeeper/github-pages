'use client';

import { useState } from 'react';
import { useAppDispatch } from '../../../../redux/store/hooks';
import { setActiveMedia } from '../../../../redux/actions/appActions';

const chapters = ['Opening title', 'Problem framing', 'Route animation', 'Results'];

export default function MediaReel() {
  const [chapter, setChapter] = useState(0);
  const dispatch = useAppDispatch();
  return <section className="rounded-2xl border border-slate-200 bg-slate-900 p-5 text-white dark:border-white/10" aria-labelledby="media-reel-title"><div className="aspect-video rounded-xl bg-gradient-to-br from-cyan-950 via-indigo-950 to-slate-950 p-5"><div className="flex h-full items-center justify-center text-center"><div><p className="text-xs font-bold uppercase tracking-[.2em] text-cyan-300">Storyboard / chapter {chapter + 1}</p><h2 id="media-reel-title" className="mt-3 text-2xl font-black">{chapters[chapter]}</h2><p className="mt-2 text-sm text-slate-300">A video slot for research explainers and game-world development logs.</p></div></div></div><div className="mt-4 flex flex-wrap gap-2" role="tablist" aria-label="Video chapters">{chapters.map((item, index) => <button key={item} type="button" role="tab" aria-selected={chapter === index} onClick={() => { setChapter(index); dispatch(setActiveMedia(`media-reel:${index}`)); }} className={`rounded-full px-3 py-1 text-xs ${chapter === index ? 'bg-cyan-300 text-slate-950' : 'bg-white/10 text-slate-300'}`}>{item}</button>)}</div></section>;
}
