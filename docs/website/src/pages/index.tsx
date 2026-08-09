import type { ReactNode } from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import styles from './index.module.css';

const CARDS: { title: string; description: string; to: string; icon: string }[] = [
  {
    icon: '📚',
    title: 'Docs',
    description: 'Architecture, ADRs, the roadmap/changelog, and every feature roadmap under docs/moon/roadmaps/.',
    to: '/docs/ARCHITECTURE',
  },
  {
    icon: '🧭',
    title: 'Guides',
    description: 'The repository README, contributing guide, and AGENTS.md — curated, not a full repo crawl.',
    to: '/guides',
  },
  {
    icon: '🧩',
    title: 'API Reference',
    description: "TypeDoc-generated reference for lib/'s exported functions and types.",
    to: '/docs/website/react/api-docs',
  },
  {
    icon: '🎨',
    title: 'Storybook',
    description: 'Every UI/layout component from src/frameworks/react/components/, rendered live with react-docgen-typescript prop tables.',
    to: '/storybook/index.html',
  },
];

function HomepageHeader() {
  const { siteConfig } = useDocusaurusContext();
  return (
    <header className={styles.heroBanner}>
      <div className={styles.heroGlow} aria-hidden="true" />
      <div className="container">
        <Heading as="h1" className={styles.heroTitle}>
          {siteConfig.title}
        </Heading>
        <p className={styles.heroSubtitle}>{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link className={styles.primaryButton} to="/docs/ARCHITECTURE">
            Read the Docs
          </Link>
          <Link className={styles.secondaryButton} to="/storybook/index.html">
            Open Storybook
          </Link>
        </div>
      </div>
    </header>
  );
}

function Cards() {
  return (
    <section className={styles.cardsSection}>
      <div className="container">
        <div className={styles.cardGrid}>
          {CARDS.map((card) => (
            <Link key={card.title} to={card.to} className={styles.card}>
              <span className={styles.cardIcon} aria-hidden="true">
                {card.icon}
              </span>
              <Heading as="h3" className={styles.cardTitle}>
                {card.title}
              </Heading>
              <p className={styles.cardDescription}>{card.description}</p>
            </Link>
          ))}
        </div>
      </div>
    </section>
  );
}

export default function Home(): ReactNode {
  const { siteConfig } = useDocusaurusContext();
  return (
    <Layout
      title={siteConfig.title}
      description="Documentation, guides, API reference, and a live component library for ACFHarbinger's personal site."
    >
      <HomepageHeader />
      <main>
        <Cards />
      </main>
    </Layout>
  );
}
