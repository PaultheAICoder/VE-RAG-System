import { useEffect, useCallback } from 'react';
import { useUIStore } from '../stores/uiStore';

interface KeyboardShortcutsOptions {
  onFocusSearch?: () => void;
}

/**
 * Global keyboard shortcuts hook
 *
 * Shortcuts:
 * - Cmd/Ctrl + K: Focus search
 * - Cmd/Ctrl + D: Toggle dark mode
 * - Cmd/Ctrl + / or ?: Show keyboard shortcuts help
 * - Escape: Close modals (handled by Modal component)
 *
 * Note: Cmd/Ctrl + N for new session is handled locally by ChatView
 * Note: Cmd/Ctrl + Enter for send is handled locally by ChatInput
 */
export function useKeyboardShortcuts(options: KeyboardShortcutsOptions = {}) {
  const { onFocusSearch } = options;
  const { toggleDarkMode, toggleShortcutsModal } = useUIStore();

  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      const isMod = event.metaKey || event.ctrlKey;
      const isShift = event.shiftKey;
      const target = event.target as HTMLElement;
      const tagName = target.tagName.toLowerCase();

      // Don't trigger shortcuts when typing in input fields (except for specific shortcuts)
      const isInputField =
        tagName === 'input' ||
        tagName === 'textarea' ||
        target.isContentEditable;

      // Cmd/Ctrl + K: Focus search
      if (isMod && event.key === 'k') {
        event.preventDefault();
        if (onFocusSearch) {
          onFocusSearch();
        } else {
          // Default: focus the first search input on the page
          const searchInput = document.querySelector<HTMLInputElement>(
            'input[placeholder*="Search"], input[type="search"]'
          );
          if (searchInput) {
            searchInput.focus();
            searchInput.select();
          }
        }
        return;
      }

      // Cmd/Ctrl + D: Toggle dark mode
      if (isMod && event.key === 'd') {
        event.preventDefault();
        toggleDarkMode();
        return;
      }

      // Cmd/Ctrl + / or ?: Show keyboard shortcuts
      if ((isMod && event.key === '/') || (isShift && event.key === '?')) {
        // Allow ? in input fields
        if (isShift && event.key === '?' && isInputField) {
          return;
        }
        event.preventDefault();
        toggleShortcutsModal();
        return;
      }
    },
    [toggleDarkMode, toggleShortcutsModal, onFocusSearch]
  );

  useEffect(() => {
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [handleKeyDown]);
}

/**
 * Hook to get platform-specific modifier key display
 */
export function usePlatformModifier() {
  const isMac =
    typeof navigator !== 'undefined' &&
    /Mac|iPod|iPhone|iPad/.test(navigator.platform);

  return {
    mod: isMac ? 'Cmd' : 'Ctrl',
    modSymbol: isMac ? '\u2318' : 'Ctrl',
  };
}
