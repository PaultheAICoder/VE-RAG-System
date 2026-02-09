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
    <header className="sticky top-0 z-50 bg-white/80 dark:bg-[#0A0A0B]/80 glass border-b border-gray-100 dark:border-[#1E1E22]">
      <div className="max-w-7xl mx-auto px-6 h-14 flex items-center justify-between">
        {/* Logo */}
        <div className="flex items-center gap-2.5">
          <span className="text-primary">
            <BridgeLogo size={40} />
          </span>
          <span className="font-semibold text-[13px] text-gray-900 dark:text-white tracking-[0.08em] uppercase">
            AI Ready RAG
          </span>
        </div>

        {/* Right side controls */}
        <div className="flex items-center gap-1">
          {/* Keyboard shortcuts */}
          <button
            onClick={toggleShortcutsModal}
            className="p-2 rounded-lg text-gray-400 dark:text-gray-500 hover:text-gray-600 dark:hover:text-gray-300 hover:bg-gray-50 dark:hover:bg-white/[0.04] transition-all duration-150"
            title={`Keyboard shortcuts (${modSymbol}+/)`}
          >
            <Keyboard size={18} />
          </button>

          {/* Dark mode toggle */}
          <button
            onClick={toggleDarkMode}
            className="p-2 rounded-lg text-gray-400 dark:text-gray-500 hover:text-gray-600 dark:hover:text-gray-300 hover:bg-gray-50 dark:hover:bg-white/[0.04] transition-all duration-150"
            title={`${darkMode ? 'Switch to light mode' : 'Switch to dark mode'} (${modSymbol}+D)`}
          >
            {darkMode ? <Sun size={18} /> : <Moon size={18} />}
          </button>

          {/* Separator */}
          <div className="w-px h-5 bg-gray-200 dark:bg-[#1E1E22] mx-2" />

          {/* User menu */}
          <div className="flex items-center gap-2 text-sm">
            <div className="w-7 h-7 rounded-full bg-primary/10 dark:bg-primary/20 flex items-center justify-center">
              <User size={14} className="text-primary" />
            </div>
            <span className="text-gray-600 dark:text-gray-400 text-[13px] font-medium">
              {user?.display_name}
            </span>
            <button
              onClick={() => logout()}
              className="p-2 rounded-lg text-gray-400 dark:text-gray-500 hover:text-gray-600 dark:hover:text-gray-300 hover:bg-gray-50 dark:hover:bg-white/[0.04] transition-all duration-150"
              title="Logout"
            >
              <LogOut size={16} />
            </button>
          </div>
        </div>
      </div>
    </header>
  );
}
