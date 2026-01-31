import { apiClient } from './client';
import type { UserWithTags, UserCreate, UserUpdate } from '../types';

/**
 * List all users (admin only).
 */
export async function listUsers(skip = 0, limit = 100): Promise<UserWithTags[]> {
  return apiClient.get<UserWithTags[]>(`/api/users?skip=${skip}&limit=${limit}`);
}

/**
 * Get a single user by ID.
 */
export async function getUser(userId: string): Promise<UserWithTags> {
  return apiClient.get<UserWithTags>(`/api/users/${userId}`);
}

/**
 * Create a new user (admin only).
 */
export async function createUser(data: UserCreate): Promise<UserWithTags> {
  return apiClient.post<UserWithTags>('/api/users', data);
}

/**
 * Update a user (admin only).
 */
export async function updateUser(userId: string, data: UserUpdate): Promise<UserWithTags> {
  return apiClient.put<UserWithTags>(`/api/users/${userId}`, data);
}

/**
 * Deactivate a user (soft delete, admin only).
 */
export async function deactivateUser(userId: string): Promise<{ message: string }> {
  return apiClient.delete<{ message: string }>(`/api/users/${userId}`);
}

/**
 * Assign tags to a user (admin only).
 */
export async function assignUserTags(
  userId: string,
  tagIds: string[]
): Promise<{ message: string; tags: { id: string; name: string }[] }> {
  return apiClient.post<{ message: string; tags: { id: string; name: string }[] }>(
    `/api/users/${userId}/tags`,
    { tag_ids: tagIds }
  );
}

/**
 * Reset user password (admin only). Returns temporary password.
 */
export async function resetUserPassword(
  userId: string
): Promise<{ temporary_password: string; message: string }> {
  return apiClient.post<{ temporary_password: string; message: string }>(
    `/api/users/${userId}/reset-password`
  );
}
