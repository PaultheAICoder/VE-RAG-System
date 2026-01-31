import { apiClient } from './client';
import type { HealthResponse, DetailedHealthResponse } from '../types';

/**
 * Get basic health status.
 */
export async function getHealth(): Promise<HealthResponse> {
  return apiClient.get<HealthResponse>('/api/health');
}

/**
 * Get detailed health status for admin dashboard.
 * Requires admin authentication.
 */
export async function getDetailedHealth(): Promise<DetailedHealthResponse> {
  return apiClient.get<DetailedHealthResponse>('/api/admin/health/detailed');
}

/**
 * Get version info.
 */
export async function getVersion(): Promise<{ name: string; version: string }> {
  return apiClient.get<{ name: string; version: string }>('/api/version');
}
