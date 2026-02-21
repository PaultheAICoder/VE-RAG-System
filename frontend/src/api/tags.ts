import { apiClient } from './client';
import type { Tag, TagFacetsResponse } from '../types';

export interface TagCreate {
  name: string;
  display_name: string;
  description?: string | null;
  color?: string;
  owner_id?: string | null;
}

export interface TagUpdate {
  display_name?: string;
  description?: string | null;
  color?: string;
  owner_id?: string | null;
}

/**
 * List all tags.
 */
export async function listTags(): Promise<Tag[]> {
  return apiClient.get<Tag[]>('/api/tags');
}

/**
 * Get a single tag by ID.
 */
export async function getTag(id: string): Promise<Tag> {
  return apiClient.get<Tag>(`/api/tags/${id}`);
}

/**
 * Create a new tag (admin only).
 */
export async function createTag(data: TagCreate): Promise<Tag> {
  return apiClient.post<Tag>('/api/tags', data);
}

/**
 * Update a tag (admin only).
 */
export async function updateTag(id: string, data: TagUpdate): Promise<Tag> {
  return apiClient.put<Tag>(`/api/tags/${id}`, data);
}

/**
 * Delete a tag (admin only).
 */
export async function deleteTag(id: string): Promise<{ message: string }> {
  return apiClient.delete<{ message: string }>(`/api/tags/${id}`);
}

/**
 * Delete ALL non-system tags (admin only). Requires confirmation.
 */
export async function deleteAllTags(): Promise<{
  deleted_count: number;
  skipped_system_count: number;
  success: boolean;
}> {
  return apiClient.delete<{
    deleted_count: number;
    skipped_system_count: number;
    success: boolean;
  }>('/api/tags', { confirm: true });
}

/**
 * Get tag facets grouped by namespace.
 */
export async function getTagFacets(): Promise<TagFacetsResponse> {
  return apiClient.get<TagFacetsResponse>('/api/tags/facets');
}
