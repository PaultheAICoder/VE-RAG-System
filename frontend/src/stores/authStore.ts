import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { apiClient } from '../api/client';
import type { User } from '../types';
import { useTenantStore } from './tenantStore';

interface AuthState {
  token: string | null;
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  login: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  checkAuth: () => Promise<void>;
  clearError: () => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      token: null,
      user: null,
      isAuthenticated: false,
      isLoading: false,
      error: null,

      login: async (email: string, password: string) => {
        set({ isLoading: true, error: null });
        try {
          const response = await apiClient.post<{ access_token: string; user: User }>('/api/auth/login', { email, password });
          const { access_token, user } = response;
          set({
            token: access_token,
            user,
            isAuthenticated: true,
            isLoading: false,
          });
          // Load tenant config after successful login
          void useTenantStore.getState().loadConfig(access_token);
        } catch (error) {
          set({
            error: error instanceof Error ? error.message : 'Login failed',
            isLoading: false,
          });
          throw error;
        }
      },

      logout: async () => {
        try {
          await apiClient.post('/api/auth/logout');
        } finally {
          set({
            token: null,
            user: null,
            isAuthenticated: false,
          });
        }
      },

      checkAuth: async () => {
        const { token } = get();
        if (!token) return;

        try {
          const user = await apiClient.get<User>('/api/auth/me');
          set({ user, isAuthenticated: true });
          // Load tenant config after session restore
          void useTenantStore.getState().loadConfig(token);
        } catch {
          set({ token: null, user: null, isAuthenticated: false });
        }
      },

      clearError: () => set({ error: null }),
    }),
    {
      name: 'auth-storage',
      partialize: (state) => ({ token: state.token }),
    }
  )
);
