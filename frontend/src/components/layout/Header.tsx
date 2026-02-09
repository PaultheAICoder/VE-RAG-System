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
    <header className="sticky top-0 z-50 bg-white/90 dark:bg-plum-900/90 backdrop-blur-md border-b border-rose-100 dark:border-mauve-700">
      <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
        {/* Logo */}
        <div className="flex items-center gap-2">
          <span className="text-primary dark:text-rose-300">
            <BridgeLogo size={48} />
          </span>
          <span className="font-bold text-sm text-rose-800 dark:text-rose-200 tracking-wide font-heading">
            AI READY RAG
          </span>
        </div>

        {/* Right side controls */}
        <div className="flex items-center gap-3">
          {/* Keyboard shortcuts */}
          <button
            onClick={toggleShortcutsModal}
            className="p-2 rounded-xl text-rose-400 dark:text-rose-300/60 hover:bg-rose-50 dark:hover:bg-mauve-800 hover:text-rose-600 dark:hover:text-rose-200 transition-all duration-200"
            title={`Keyboard shortcuts (${modSymbol}+/)`}
          >
            <Keyboard size={20} />
          </button>

          {/* Dark mode toggle */}
          <button
            onClick={toggleDarkMode}
            className="p-2 rounded-xl text-rose-400 dark:text-rose-300/60 hover:bg-rose-50 dark:hover:bg-mauve-800 hover:text-rose-600 dark:hover:text-rose-200 transition-all duration-200"
            title={`${darkMode ? 'Switch to light mode' : 'Switch to dark mode'} (${modSymbol}+D)`}
          >
            {darkMode ? <Sun size={20} /> : <Moon size={20} />}
          </button>

          {/* User menu */}
          <div className="flex items-center gap-2 text-sm pl-2 border-l border-rose-100 dark:border-mauve-700">
            <div className="w-7 h-7 rounded-full bg-gradient-to-br from-primary to-rose-500 flex items-center justify-center">
              <User size={14} className="text-white" />
            </div>
            <span className="text-rose-700 dark:text-rose-300 font-medium">
              {user?.display_name}
            </span>
            <button
              onClick={() => logout()}
              className="p-2 rounded-xl text-rose-400 dark:text-rose-300/60 hover:bg-rose-50 dark:hover:bg-mauve-800 hover:text-rose-600 dark:hover:text-rose-200 transition-all duration-200"
              title="Logout"
            >
              <LogOut size={18} />
            </button>
          </div>
        </div>
      </div>
    </header>
  );
}
