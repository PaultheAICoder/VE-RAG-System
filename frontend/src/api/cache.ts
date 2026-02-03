import { apiClient } from './client';
import { useAuthStore } from '../stores/authStore';
import type {
  CacheSettings,
  CacheStats,
  HitRateTrendPoint,
  TopCachedQuery,
  ClearCacheResponse,
  WarmCacheResponse,
  WarmFileResponse,
  WarmingJob,
  WarmingJobListResponse,
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

/**
 * Upload a file to warm cache with progress tracking via SSE.
 * Returns job_id and SSE URL for progress monitoring.
 * @param file - Text file (.txt or .csv) containing queries
 */
export async function warmCacheFromFile(file: File): Promise<WarmFileResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const token = useAuthStore.getState().token;
  const response = await fetch('/api/admin/cache/warm-file', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
    },
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Upload failed' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }
  return response.json();
}

/**
 * Retry specific failed queries from a previous warming job.
 * @param queries - Array of failed queries to retry
 */
export async function retryWarmCache(queries: string[]): Promise<WarmFileResponse> {
  const token = useAuthStore.getState().token;
  const response = await fetch('/api/admin/cache/warm-retry', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ queries }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Retry failed' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }
  return response.json();
}

/**
 * Get SSE URL for warming progress.
 * Note: Token passed as query param for EventSource compatibility.
 * @param jobId - The warming job ID
 */
export function getWarmProgressUrl(jobId: string): string {
  const token = useAuthStore.getState().token;
  return `/api/admin/cache/warm-progress/${jobId}?token=${token}`;
}

/**
 * Get current status of a warming job.
 * @param jobId - The warming job ID
 */
export async function getWarmStatus(jobId: string): Promise<{
  job_id: string;
  status: string;
  total: number;
  processed: number;
  success_count: number;
  failed_queries: string[];
  started_at: string | null;
  completed_at: string | null;
}> {
  return apiClient.get(`/api/admin/cache/warm-status/${jobId}`);
}

// =============================================================================
// Queue Management API Functions
// =============================================================================

/**
 * Get list of all warming jobs with optional status filter.
 */
export async function getWarmingJobs(status?: string): Promise<WarmingJobListResponse> {
  const url = status
    ? `/api/admin/cache/warm-jobs?status=${status}`
    : '/api/admin/cache/warm-jobs';
  return apiClient.get<WarmingJobListResponse>(url);
}

/**
 * Get the currently active (running) warming job.
 */
export async function getActiveWarmingJob(): Promise<WarmingJob | null> {
  return apiClient.get<WarmingJob | null>('/api/admin/cache/warm-jobs/active');
}

/**
 * Pause a running warming job.
 */
export async function pauseWarmingJob(jobId: string): Promise<WarmingJob> {
  return apiClient.post<WarmingJob>(`/api/admin/cache/warm-jobs/${jobId}/pause`);
}

/**
 * Resume a paused warming job.
 */
export async function resumeWarmingJob(jobId: string): Promise<WarmingJob> {
  return apiClient.post<WarmingJob>(`/api/admin/cache/warm-jobs/${jobId}/resume`);
}

/**
 * Delete a warming job (any status).
 */
export async function deleteWarmingJob(jobId: string): Promise<void> {
  await apiClient.delete(`/api/admin/cache/warm-jobs/${jobId}`);
}
