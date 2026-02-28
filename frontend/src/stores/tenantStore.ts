import { create } from 'zustand';
import { TenantConfig, DEFAULT_TENANT_CONFIG, fetchTenantConfig } from '../api/tenant';

interface TenantState {
  config: TenantConfig;
  isLoaded: boolean;
  isLoading: boolean;
  error: string | null;
  loadConfig: (token: string) => Promise<void>;
  hasFeature: (flag: keyof TenantConfig['feature_flags']) => boolean;
}

export const useTenantStore = create<TenantState>((set, get) => ({
  config: DEFAULT_TENANT_CONFIG,
  isLoaded: false,
  isLoading: false,
  error: null,

  loadConfig: async (token: string) => {
    set({ isLoading: true, error: null });
    try {
      const config = await fetchTenantConfig(token);
      set({ config, isLoaded: true, isLoading: false });

      // Apply CSS custom properties for branding
      if (config.branding?.primary_color) {
        document.documentElement.style.setProperty(
          '--color-primary',
          config.branding.primary_color
        );
      }
      if (config.branding?.product_name) {
        document.title = config.branding.product_name;
      }
    } catch (err) {
      set({
        config: DEFAULT_TENANT_CONFIG,
        isLoaded: true,
        isLoading: false,
        error: String(err),
      });
    }
  },

  hasFeature: (flag) => {
    return get().config.feature_flags[flag] === true;
  },
}));

export function useTenantConfig() {
  return useTenantStore();
}
