import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface UIState {
  darkMode: boolean;
  sidebarCollapsed: boolean;
  toggleDarkMode: () => void;
  toggleSidebar: () => void;
}

export const useUIStore = create<UIState>()(
  persist(
    (set) => ({
      darkMode: false,
      sidebarCollapsed: false,

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
    }),
    {
      name: 'ui-storage',
      onRehydrateStorage: () => (state) => {
        // Apply dark mode on rehydration
        if (state?.darkMode) {
          document.documentElement.classList.add('dark');
        }
      },
    }
  )
);
