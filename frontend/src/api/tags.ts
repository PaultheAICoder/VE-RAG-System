import { apiClient } from './client';
import type { Tag } from '../types';

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
