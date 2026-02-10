import { apiClient } from './client';
import { useAuthStore } from '../stores/authStore';
import type {
  CacheSettings,
  CacheStats,
  HitRateTrendPoint,
  TopCachedQuery,
  ClearCacheResponse,
  WarmFileResponse,
  WarmingJob,
  WarmingJobListResponse,
  BatchQueriesResponse,
  QueryRetryResponse,
  WarmingQueryStatus,
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
 * Uses the DB-based warming queue for reliable processing.
 * @param queries - Array of queries to pre-cache
 */
export async function warmCache(queries: string[]): Promise<WarmFileResponse> {
  return apiClient.post<WarmFileResponse>('/api/admin/warming/queue/manual', { queries });
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
  const response = await fetch('/api/admin/warming/queue/upload', {
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
 * Get queries for a specific warming batch.
 * @param batchId - The batch ID
 * @param statusFilter - Optional status filter
 * @param limit - Optional limit
 * @param offset - Optional offset
 */
export async function getBatchQueries(
  batchId: string,
  statusFilter?: WarmingQueryStatus,
  limit?: number,
  offset?: number,
): Promise<BatchQueriesResponse> {
  let url = `/api/admin/warming/batch/${batchId}/queries`;
  const params = new URLSearchParams();
  if (statusFilter) params.set('status', statusFilter);
  if (limit) params.set('limit', String(limit));
  if (offset) params.set('offset', String(offset));
  const qs = params.toString();
  if (qs) url += `?${qs}`;
  return apiClient.get<BatchQueriesResponse>(url);
}

/**
 * Retry all failed queries in a batch.
 * @param batchId - The batch ID to retry
 */
export async function retryBatch(batchId: string): Promise<QueryRetryResponse> {
  return apiClient.post<QueryRetryResponse>(
    `/api/admin/warming/batch/${batchId}/retry`
  );
}

/**
 * Retry a single failed query.
 * @param batchId - The batch ID
 * @param queryId - The query ID to retry
 */
export async function retryQuery(
  batchId: string,
  queryId: string,
): Promise<QueryRetryResponse> {
  return apiClient.post<QueryRetryResponse>(
    `/api/admin/warming/batch/${batchId}/queries/${queryId}/retry`
  );
}

/**
 * Get SSE URL for warming progress.
 * Note: Token passed as query param for EventSource compatibility.
 * @param jobId - The warming job ID
 * @param lastEventId - Optional last event ID for resumption
 */
export function getWarmProgressUrl(jobId: string, lastEventId?: string | null): string {
  const token = useAuthStore.getState().token;
  let url = `/api/admin/warming/progress?job_id=${jobId}&token=${token}`;
  if (lastEventId) {
    url += `&last_event_id=${lastEventId}`;
  }
  return url;
}

/**
 * Get current status of a warming job.
 * @param jobId - The warming job ID
 */
export async function getWarmStatus(jobId: string): Promise<WarmingJob> {
  return apiClient.get<WarmingJob>(`/api/admin/warming/queue/${jobId}`);
}

// =============================================================================
// Queue Management API Functions
// =============================================================================

/**
 * Get list of all warming jobs with optional status filter.
 */
export async function getWarmingJobs(status?: string): Promise<WarmingJobListResponse> {
  const url = status
    ? `/api/admin/warming/queue?status=${status}`
    : '/api/admin/warming/queue';
  return apiClient.get<WarmingJobListResponse>(url);
}

/**
 * Get the currently active (running) warming job.
 */
export async function getActiveWarmingJob(): Promise<WarmingJob | null> {
  return apiClient.get<WarmingJob | null>('/api/admin/warming/current');
}

/**
 * Pause the currently running warming job.
 * @param _jobId - Unused (kept for API compatibility)
 */
// eslint-disable-next-line @typescript-eslint/no-unused-vars
export async function pauseWarmingJob(_jobId: string): Promise<WarmingJob> {
  return apiClient.post<WarmingJob>('/api/admin/warming/current/pause');
}

/**
 * Resume the paused warming job.
 * @param _jobId - Unused (kept for API compatibility)
 */
// eslint-disable-next-line @typescript-eslint/no-unused-vars
export async function resumeWarmingJob(_jobId: string): Promise<WarmingJob> {
  return apiClient.post<WarmingJob>('/api/admin/warming/current/resume');
}

/**
 * Delete a warming job (any status).
 */
export async function deleteWarmingJob(jobId: string): Promise<void> {
  await apiClient.delete(`/api/admin/warming/queue/${jobId}`);
}

/**
 * Cancel the currently running warming job.
 */
export async function cancelWarmingJob(): Promise<WarmingJob> {
  return apiClient.post<WarmingJob>('/api/admin/warming/current/cancel');
}

/**
 * Bulk delete multiple warming jobs.
 * @param jobIds - Array of job IDs to delete
 */
export async function bulkDeleteWarmingJobs(jobIds: string[]): Promise<{ deleted_count: number }> {
  return apiClient.delete<{ deleted_count: number }>('/api/admin/warming/queue/bulk', {
    job_ids: jobIds,
  });
}

/**
 * Add manual queries to the warming queue.
 * @param queries - Array of queries to add
 */
export async function addManualQueries(queries: string[]): Promise<WarmFileResponse> {
  return apiClient.post<WarmFileResponse>('/api/admin/warming/queue/manual', { queries });
}
