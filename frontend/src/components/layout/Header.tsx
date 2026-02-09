import { Moon, Sun, LogOut, User, Keyboard } from 'lucide-react';
import { useAuthStore } from '../../stores/authStore';
import { useUIStore } from '../../stores/uiStore';
import { BridgeLogo } from './BridgeLogo';
import { usePlatformModifier } from '../../hooks';

export function Header() {
  const { user, logout } = useAuthStore();
  const { darkMode, toggleDarkMode, toggleShortcutsModal } = useUIStore();
  const { modSymbol } = usePlatformModifier();

  return (
    <header className="sticky top-0 z-50 bg-white/95 dark:bg-warm-800/95 backdrop-blur-md border-b border-warm-200 dark:border-warm-700">
      <div className="max-w-7xl mx-auto px-5 py-3 flex items-center justify-between">
        {/* Logo */}
        <div className="flex items-center gap-2.5">
          <span className="text-primary">
            <BridgeLogo size={48} />
          </span>
          <span className="font-heading font-semibold text-sm text-warm-900 dark:text-cream tracking-wide">
            AI READY RAG
          </span>
        </div>

        {/* Right side controls */}
        <div className="flex items-center gap-1">
          {/* Keyboard shortcuts */}
          <button
            onClick={toggleShortcutsModal}
            className="p-2 rounded-xl text-warm-500 dark:text-warm-400 hover:text-warm-700 dark:hover:text-warm-300 hover:bg-warm-100 dark:hover:bg-warm-700 transition-colors"
            title={`Keyboard shortcuts (${modSymbol}+/)`}
          >
            <Keyboard size={19} />
          </button>

          {/* Dark mode toggle */}
          <button
            onClick={toggleDarkMode}
            className="p-2 rounded-xl text-warm-500 dark:text-warm-400 hover:text-warm-700 dark:hover:text-warm-300 hover:bg-warm-100 dark:hover:bg-warm-700 transition-colors"
            title={`${darkMode ? 'Switch to light mode' : 'Switch to dark mode'} (${modSymbol}+D)`}
          >
            {darkMode ? <Sun size={19} /> : <Moon size={19} />}
          </button>

          {/* Separator */}
          <div className="w-px h-5 bg-warm-200 dark:bg-warm-700 mx-2" />

          {/* User menu */}
          <div className="flex items-center gap-2 text-sm">
            <div className="flex items-center justify-center w-7 h-7 rounded-full bg-primary/10 dark:bg-primary/15">
              <User size={15} className="text-primary" />
            </div>
            <span className="text-warm-700 dark:text-warm-300 font-medium">
              {user?.display_name}
            </span>
            <button
              onClick={() => logout()}
              className="p-2 rounded-xl text-warm-500 dark:text-warm-400 hover:text-warm-700 dark:hover:text-warm-300 hover:bg-warm-100 dark:hover:bg-warm-700 transition-colors"
              title="Logout"
            >
              <LogOut size={17} />
            </button>
          </div>
        </div>
      </div>
    </header>
  );
}
