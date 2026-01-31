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
    <header className="sticky top-0 z-50 bg-white/95 dark:bg-gray-900/95 backdrop-blur border-b border-gray-200 dark:border-gray-800">
      <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
        {/* Logo */}
        <div className="flex items-center gap-2">
          <span className="text-primary">
            <BridgeLogo size={48} />
          </span>
          <span className="font-bold text-sm text-gray-900 dark:text-white tracking-wide">
            AI READY RAG
          </span>
        </div>

        {/* Right side controls */}
        <div className="flex items-center gap-4">
          {/* Keyboard shortcuts */}
          <button
            onClick={toggleShortcutsModal}
            className="p-2 rounded-lg text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
            title={`Keyboard shortcuts (${modSymbol}+/)`}
          >
            <Keyboard size={20} />
          </button>

          {/* Dark mode toggle */}
          <button
            onClick={toggleDarkMode}
            className="p-2 rounded-lg text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
            title={`${darkMode ? 'Switch to light mode' : 'Switch to dark mode'} (${modSymbol}+D)`}
          >
            {darkMode ? <Sun size={20} /> : <Moon size={20} />}
          </button>

          {/* User menu */}
          <div className="flex items-center gap-2 text-sm">
            <User size={18} className="text-gray-400" />
            <span className="text-gray-700 dark:text-gray-300">
              {user?.display_name}
            </span>
            <button
              onClick={() => logout()}
              className="p-2 rounded-lg text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
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
