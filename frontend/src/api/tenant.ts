/**
 * Tenant configuration API client.
 * Fetches tenant config from GET /api/tenant/config
 */

export interface AIModelConfig {
  enrichment_model: string;
  query_model_simple: string;
  query_model_complex: string;
}

export interface FeatureFlags {
  ca_enabled: boolean;
  ca_auto_renewal: boolean;
  ca_unit_owner_letters: boolean;
  ca_board_presentations: boolean;
  structured_query_enabled: boolean;
  claude_enrichment_enabled: boolean;
}

export interface BrandingConfig {
  primary_color: string;
  logo_url: string | null;
  product_name: string;
}

export interface TenantConfig {
  tenant_id: string;
  display_name: string;
  active_modules: string[];
  feature_flags: FeatureFlags;
  ai_models: AIModelConfig;
  branding: BrandingConfig;
}

const DEFAULT_TENANT_CONFIG: TenantConfig = {
  tenant_id: 'default',
  display_name: '',
  active_modules: ['core'],
  feature_flags: {
    ca_enabled: false,
    ca_auto_renewal: false,
    ca_unit_owner_letters: false,
    ca_board_presentations: false,
    structured_query_enabled: false,
    claude_enrichment_enabled: false,
  },
  ai_models: {
    enrichment_model: 'claude-sonnet-4-6',
    query_model_simple: 'claude-haiku-4-5-20251001',
    query_model_complex: 'claude-sonnet-4-6',
  },
  branding: {
    primary_color: '#1a73e8',
    logo_url: null,
    product_name: 'VaultIQ',
  },
};

export async function fetchTenantConfig(token: string): Promise<TenantConfig> {
  const baseUrl = import.meta.env.VITE_API_URL || '';
  const response = await fetch(`${baseUrl}/api/tenant/config`, {
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    console.warn('Failed to load tenant config, using defaults', response.status);
    return DEFAULT_TENANT_CONFIG;
  }

  return response.json();
}

export { DEFAULT_TENANT_CONFIG };
