// @ts-check
import {themes as prismThemes} from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Physical AI Textbook',
  tagline: 'Learn AI & Robotics Step by Step',
  favicon: 'img/favicon.ico',
  url: 'http://localhost:3000',
  baseUrl: '/',
  onBrokenLinks: 'throw',
  organizationName: 'my-org', // GitHub org
  projectName: 'physical-ai-textbook', // Repo name
  i18n: { defaultLocale: 'en', locales: ['en'] },

  presets: [
    [
      'classic',
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl: '#', // Optional
        },
        blog: {
          showReadingTime: true,
          editUrl: '#',
        },
        theme: { customCss: require.resolve('./src/css/custom.css') },
      }),
    ],
  ],

  themeConfig: {
    image: 'img/docusaurus-social-card.jpg',
    colorMode: { respectPrefersColorScheme: true },
    navbar: {
      title: 'Physical AI Textbook',
      logo: { alt: 'Logo', src: 'img/logo.svg' },
      items: [
        { type: 'docSidebar', sidebarId: 'tutorialSidebar', position: 'left', label: 'Chapters' },
        { to: '/blog', label: 'Blog', position: 'left' },
        { href: 'https://github.com/facebook/docusaurus', label: 'GitHub', position: 'right' },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        { title: 'Docs', items: [{ label: 'Chapters', to: '/docs/introduction' }] },
        {
          title: 'Community',
          items: [
            { label: 'Stack Overflow', href: 'https://stackoverflow.com' },
            { label: 'Discord', href: 'https://discord.com' },
            { label: 'X', href: 'https://x.com' },
          ],
        },
        { title: 'More', items: [{ label: 'Blog', to: '/blog' }, { label: 'GitHub', href: 'https://github.com/facebook/docusaurus' }] },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} My Project.`,
    },
    prism: { theme: prismThemes.github, darkTheme: prismThemes.dracula },
  },
};

export default config;
