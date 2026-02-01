import { apiClient } from './client';
import type {
  CacheSettings,
  CacheStats,
  HitRateTrendPoint,
  TopCachedQuery,
  ClearCacheResponse,
  WarmCacheResponse,
} from '../types';

/**
 * Get current cache settings (system admin only).
 */
export async function getCacheSettings(): Promise<CacheSettings> {
  return apiClient.get<CacheSettings>('/api/admin/cache/settings');
}

/**
 * Update cache settings (system admin only).
 */
export async function updateCacheSettings(
  settings: Partial<CacheSettings>
): Promise<CacheSettings> {
  return apiClient.put<CacheSettings>('/api/admin/cache/settings', settings);
}

/**
 * Get cache statistics (system admin only).
 */
export async function getCacheStats(): Promise<CacheStats> {
  return apiClient.get<CacheStats>('/api/admin/cache/stats');
}

/**
 * Clear all cache entries (system admin only).
 */
export async function clearCache(): Promise<ClearCacheResponse> {
  return apiClient.post<ClearCacheResponse>('/api/admin/cache/clear', {});
}

/**
 * Get top cached queries (system admin only).
 * @param limit - Maximum number of queries to return (default: 10)
 */
export async function getTopQueries(limit?: number): Promise<{ queries: TopCachedQuery[] }> {
  const url = limit
    ? `/api/admin/cache/top-queries?limit=${limit}`
    : '/api/admin/cache/top-queries';
  return apiClient.get<{ queries: TopCachedQuery[] }>(url);
}

/**
 * Get hit rate trend data (system admin only).
 * @param days - Number of days to include (default: 7)
 */
export async function getHitRateTrend(days?: number): Promise<{ trend: HitRateTrendPoint[] }> {
  const url = days
    ? `/api/admin/cache/trend?days=${days}`
    : '/api/admin/cache/trend';
  return apiClient.get<{ trend: HitRateTrendPoint[] }>(url);
}

/**
 * Warm cache with queries (system admin only).
 * @param queries - Array of queries to pre-cache
 */
export async function warmCache(queries: string[]): Promise<WarmCacheResponse> {
  return apiClient.post<WarmCacheResponse>('/api/admin/cache/warm', { queries });
}
