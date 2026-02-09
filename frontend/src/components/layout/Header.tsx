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
    <header className="sticky top-0 z-50 corp-header-gradient border-b border-corp-blue-dark dark:border-corp-navy">
      <div className="max-w-7xl mx-auto px-4 py-2.5 flex items-center justify-between">
        {/* Logo */}
        <div className="flex items-center gap-2">
          <span className="text-white">
            <BridgeLogo size={48} />
          </span>
          <div className="flex flex-col">
            <span className="font-bold text-sm text-white tracking-wide font-heading" style={{ textShadow: '0 1px 2px rgba(0,0,0,0.4)' }}>
              AI READY RAG
            </span>
            <span className="text-[10px] text-blue-200 dark:text-blue-300 tracking-wider font-body">
              ENTERPRISE KNOWLEDGE SYSTEM
            </span>
          </div>
        </div>

        {/* Right side controls */}
        <div className="flex items-center gap-2">
          {/* Keyboard shortcuts */}
          <button
            onClick={toggleShortcutsModal}
            className="p-1.5 text-blue-200 hover:text-white hover:bg-white/10 transition-colors border border-transparent hover:border-blue-300/30"
            title={`Keyboard shortcuts (${modSymbol}+/)`}
          >
            <Keyboard size={18} />
          </button>

          {/* Dark mode toggle */}
          <button
            onClick={toggleDarkMode}
            className="p-1.5 text-blue-200 hover:text-white hover:bg-white/10 transition-colors border border-transparent hover:border-blue-300/30"
            title={`${darkMode ? 'Switch to light mode' : 'Switch to dark mode'} (${modSymbol}+D)`}
          >
            {darkMode ? <Sun size={18} /> : <Moon size={18} />}
          </button>

          {/* Separator */}
          <div className="w-px h-5 bg-blue-400/30 mx-1"></div>

          {/* User menu */}
          <div className="flex items-center gap-2 text-sm">
            <User size={16} className="text-blue-200" />
            <span className="text-white text-xs font-body">
              {user?.display_name}
            </span>
            <button
              onClick={() => logout()}
              className="p-1.5 text-blue-200 hover:text-white hover:bg-white/10 transition-colors border border-transparent hover:border-blue-300/30"
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
