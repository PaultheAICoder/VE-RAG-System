import { apiClient } from './client';
import type { HealthResponse } from '../types';

/**
 * Get basic health status.
 */
export async function getHealth(): Promise<HealthResponse> {
  return apiClient.get<HealthResponse>('/api/health');
}

/**
 * Get version info.
 */
export async function getVersion(): Promise<{ name: string; version: string }> {
  return apiClient.get<{ name: string; version: string }>('/api/version');
}
