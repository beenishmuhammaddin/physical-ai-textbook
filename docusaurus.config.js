// @ts-check
import {themes as prismThemes} from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Physical AI Textbook',
  tagline: 'Learn AI & Robotics Step by Step',
  favicon: 'img/favicon.ico',
  url: 'http://localhost:3000',
  baseUrl: '/',
  onBrokenLinks: 'warn', // warning instead of throw to avoid build fail
  organizationName: 'my-org', // GitHub org
  projectName: 'physical-ai-textbook', // Repo name
  i18n: { defaultLocale: 'en', locales: ['en'] },

  clientModules: [
    require.resolve('./src/clientModules/chatWidget.js'),
  ],

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
        {
          title: 'Docs',
          items: [
            { label: 'Chapter 1', to: '/docs/chapter1-introduction' },
            { label: 'Chapter 2', to: '/docs/chapter2-physical-ai' },
            { label: 'Chapter 3', to: '/docs/chapter3-humanoid-robotics' },
            { label: 'Chapter 4', to: '/docs/chapter4-sensors-actuators' },
            { label: 'Chapter 5', to: '/docs/chapter5-control-systems' },
            { label: 'Chapter 6', to: '/docs/chapter6-ai-techniques' },
            { label: 'Chapter 7', to: '/docs/chapter7-applications' },
            { label: 'Chapter 8', to: '/docs/chapter8-conclusion' },
          ],
        },
        {
          title: 'Community',
          items: [
            { label: 'Stack Overflow', href: 'https://stackoverflow.com' },
            { label: 'Discord', href: 'https://discord.com' },
            { label: 'X', href: 'https://x.com' },
          ],
        },
        {
          title: 'More',
          items: [
            { label: 'Blog', to: '/blog' },
            { label: 'GitHub', href: 'https://github.com/facebook/docusaurus' },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} My Project.`,
    },

    prism: { theme: prismThemes.github, darkTheme: prismThemes.dracula },
  },
};

export default config;
