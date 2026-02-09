import { Moon, Sun, LogOut, User, Keyboard } from 'lucide-react';
import { useAuthStore } from '../../stores/authStore';
import { useUIStore } from '../../stores/uiStore';
import { usePlatformModifier } from '../../hooks';

export function Header() {
  const { user, logout } = useAuthStore();
  const { darkMode, toggleDarkMode, toggleShortcutsModal } = useUIStore();
  const { modSymbol } = usePlatformModifier();

  return (
    <header className="sticky top-0 z-50 bg-white dark:bg-[#202123] border-b border-[#E5E5E5] dark:border-[#4E4F60]">
      <div className="max-w-7xl mx-auto px-4 py-2.5 flex items-center justify-between">
        {/* Logo */}
        <div className="flex items-center gap-2.5">
          <div className="flex items-center justify-center w-8 h-8 rounded-sm bg-[#10A37F]">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 2L2 7l10 5 10-5-10-5z" />
              <path d="M2 17l10 5 10-5" />
              <path d="M2 12l10 5 10-5" />
            </svg>
          </div>
          <span className="font-semibold text-sm text-[#2D2D2D] dark:text-[#ECECF1]">
            AI Ready RAG
          </span>
        </div>

        {/* Right side controls */}
        <div className="flex items-center gap-1">
          {/* Keyboard shortcuts */}
          <button
            onClick={toggleShortcutsModal}
            className="p-2 rounded-md text-[#6E6E80] dark:text-[#ACACBE] hover:bg-[#F7F7F8] dark:hover:bg-[#2A2B32] transition-colors"
            title={`Keyboard shortcuts (${modSymbol}+/)`}
          >
            <Keyboard size={18} />
          </button>

          {/* Dark mode toggle */}
          <button
            onClick={toggleDarkMode}
            className="p-2 rounded-md text-[#6E6E80] dark:text-[#ACACBE] hover:bg-[#F7F7F8] dark:hover:bg-[#2A2B32] transition-colors"
            title={`${darkMode ? 'Switch to light mode' : 'Switch to dark mode'} (${modSymbol}+D)`}
          >
            {darkMode ? <Sun size={18} /> : <Moon size={18} />}
          </button>

          {/* Divider */}
          <div className="w-px h-5 bg-[#E5E5E5] dark:bg-[#4E4F60] mx-2" />

          {/* User menu */}
          <div className="flex items-center gap-2 text-sm">
            <User size={16} className="text-[#6E6E80] dark:text-[#ACACBE]" />
            <span className="text-[#2D2D2D] dark:text-[#ECECF1] text-sm">
              {user?.display_name}
            </span>
            <button
              onClick={() => logout()}
              className="p-2 rounded-md text-[#6E6E80] dark:text-[#ACACBE] hover:bg-[#F7F7F8] dark:hover:bg-[#2A2B32] transition-colors"
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
