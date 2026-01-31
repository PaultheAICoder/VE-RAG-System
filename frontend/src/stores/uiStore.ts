import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface UIState {
  darkMode: boolean;
  sidebarCollapsed: boolean;
  shortcutsModalOpen: boolean;
  toggleDarkMode: () => void;
  toggleSidebar: () => void;
  toggleShortcutsModal: () => void;
  setShortcutsModalOpen: (open: boolean) => void;
  initDarkMode: () => void;
}

export const useUIStore = create<UIState>()(
  persist(
    (set, get) => ({
      darkMode: false,
      sidebarCollapsed: false,
      shortcutsModalOpen: false,

      toggleDarkMode: () => {
        set((state) => {
          const newDarkMode = !state.darkMode;
          if (newDarkMode) {
            document.documentElement.classList.add('dark');
          } else {
            document.documentElement.classList.remove('dark');
          }
          return { darkMode: newDarkMode };
        });
      },

      toggleSidebar: () => {
        set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed }));
      },

      toggleShortcutsModal: () => {
        set((state) => ({ shortcutsModalOpen: !state.shortcutsModalOpen }));
      },

      setShortcutsModalOpen: (open: boolean) => {
        set({ shortcutsModalOpen: open });
      },

      initDarkMode: () => {
        // Check if we already have a stored preference (handled by persist middleware)
        const currentState = get();
        const stored = localStorage.getItem('ui-storage');

        if (stored) {
          // If there's a stored preference, persist middleware has already applied it
          // Just ensure the DOM reflects the stored state
          if (currentState.darkMode) {
            document.documentElement.classList.add('dark');
          } else {
            document.documentElement.classList.remove('dark');
          }
        } else {
          // No stored preference - check system preference
          const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
          if (prefersDark) {
            document.documentElement.classList.add('dark');
            set({ darkMode: true });
          }
        }
      },
    }),
    {
      name: 'ui-storage',
      partialize: (state) => ({
        darkMode: state.darkMode,
        sidebarCollapsed: state.sidebarCollapsed
      }),
      onRehydrateStorage: () => (state) => {
        // Apply dark mode on rehydration
        if (state?.darkMode) {
          document.documentElement.classList.add('dark');
        }
      },
    }
  )
);
