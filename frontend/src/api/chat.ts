import { apiClient } from './client';
import type { DeleteSessionResponse, BulkDeleteSessionsResponse } from '../types';

/**
 * Delete a single chat session (admin only).
 * Permanently removes the session and all its messages.
 */
export async function deleteSession(sessionId: string): Promise<DeleteSessionResponse> {
  return apiClient.delete<DeleteSessionResponse>(`/api/chat/sessions/${sessionId}`);
}

/**
 * Bulk delete multiple chat sessions (admin only).
 * Maximum 100 sessions per request.
 */
export async function bulkDeleteSessions(
  sessionIds: string[]
): Promise<BulkDeleteSessionsResponse> {
  return apiClient.delete<BulkDeleteSessionsResponse>('/api/chat/sessions/bulk', {
    session_ids: sessionIds,
  });
}
