import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import ClientLayoutWrapper from './ClientLayoutWrapper';
import nextConfig from '../next.config.js'; // Note: This path might need fixing depending on how Next.js handles it

const inter = Inter({ subsets: ['latin'] });

const basePath = nextConfig.basePath || ''; 

export const metadata: Metadata = {
  title: 'ACF | Harbinger',
  description: 'Personal blog and portfolio.',
  icons: {
    icon: `${basePath}/favicon.ico`
  }
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <ClientLayoutWrapper>{children}</ClientLayoutWrapper>
      </body>
    </html>
  );
}
