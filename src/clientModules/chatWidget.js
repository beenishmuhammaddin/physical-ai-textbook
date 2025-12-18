import React from 'react';
import { createRoot } from 'react-dom/client';
import ChatWidget from '../components/ChatWidget';
import ExecutionEnvironment from '@docusaurus/ExecutionEnvironment';

if (ExecutionEnvironment.canUseDOM) {
  // Wait for DOM to be ready
  window.addEventListener('DOMContentLoaded', () => {
    // Create container for chat widget
    const container = document.createElement('div');
    container.id = 'chat-widget-root';
    document.body.appendChild(container);

    // Render chat widget
    const root = createRoot(container);
    root.render(<ChatWidget />);
  });
}

export default function clientModule() {
  // This function is required but can be empty
}
