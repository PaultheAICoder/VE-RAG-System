import { useAuthStore } from '../stores/authStore';
import type { Tag } from '../types';

const getAuthHeaders = (): HeadersInit => {
  const token = useAuthStore.getState().token;
  const headers: HeadersInit = {};
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  return headers;
};

/**
 * List all tags.
 */
export async function listTags(): Promise<Tag[]> {
  const response = await fetch('/api/tags', {
    headers: getAuthHeaders(),
  });

  if (!response.ok) {
    if (response.status === 401) {
      useAuthStore.getState().logout();
    }
    const error = await response.json().catch(() => ({ detail: 'Failed to fetch tags' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

/**
 * Get a single tag by ID.
 */
export async function getTag(id: string): Promise<Tag> {
  const response = await fetch(`/api/tags/${id}`, {
    headers: getAuthHeaders(),
  });

  if (!response.ok) {
    if (response.status === 401) {
      useAuthStore.getState().logout();
    }
    const error = await response.json().catch(() => ({ detail: 'Failed to fetch tag' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}
