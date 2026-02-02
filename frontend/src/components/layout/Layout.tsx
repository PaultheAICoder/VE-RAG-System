import type { ReactNode } from 'react';
import { Header } from './Header';
import { Nav } from './Nav';
import { useUIStore } from '../../stores/uiStore';
import { KeyboardShortcutsModal } from '../ui';
import { useKeyboardShortcuts } from '../../hooks';

interface LayoutProps {
  children: ReactNode;
}

export function Layout({ children }: LayoutProps) {
  const { shortcutsModalOpen, setShortcutsModalOpen } = useUIStore();

  // Initialize global keyboard shortcuts
  // Note: ChatView registers its own handler for Cmd+N to create new sessions
  useKeyboardShortcuts();

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-950">
      <Header />
      <Nav />
      <main className="max-w-7xl mx-auto px-4 py-6">
        {children}
      </main>

      {/* Keyboard Shortcuts Modal */}
      <KeyboardShortcutsModal
        isOpen={shortcutsModalOpen}
        onClose={() => setShortcutsModalOpen(false)}
      />
    </div>
  );
}
