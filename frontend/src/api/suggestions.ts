import { useAuthStore } from '../stores/authStore';
import type { TagSuggestionListResponse, ApprovalResponse } from '../types';

const getAuthHeaders = (): HeadersInit => {
  const token = useAuthStore.getState().token;
  const headers: HeadersInit = {};
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  return headers;
};

/**
 * List tag suggestions for a document, optionally filtered by status.
 */
export async function listTagSuggestions(
  documentId: string,
  statusFilter?: string
): Promise<TagSuggestionListResponse> {
  const params = new URLSearchParams();
  if (statusFilter) {
    params.set('status_filter', statusFilter);
  }

  const queryString = params.toString();
  const url = `/api/documents/${documentId}/tag-suggestions${queryString ? `?${queryString}` : ''}`;

  const response = await fetch(url, {
    headers: getAuthHeaders(),
  });

  if (!response.ok) {
    if (response.status === 401) {
      useAuthStore.getState().logout();
    }
    const error = await response.json().catch(() => ({ detail: 'Failed to fetch suggestions' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

/**
 * Approve selected tag suggestions for a document.
 */
export async function approveSuggestions(
  documentId: string,
  suggestionIds: string[]
): Promise<ApprovalResponse> {
  const response = await fetch(`/api/documents/${documentId}/tag-suggestions/approve`, {
    method: 'POST',
    headers: {
      ...getAuthHeaders(),
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ suggestion_ids: suggestionIds }),
  });

  if (!response.ok) {
    if (response.status === 401) {
      useAuthStore.getState().logout();
    }
    const error = await response.json().catch(() => ({ detail: 'Failed to approve suggestions' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

/**
 * Reject selected tag suggestions for a document.
 */
export async function rejectSuggestions(
  documentId: string,
  suggestionIds: string[]
): Promise<ApprovalResponse> {
  const response = await fetch(`/api/documents/${documentId}/tag-suggestions/reject`, {
    method: 'POST',
    headers: {
      ...getAuthHeaders(),
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ suggestion_ids: suggestionIds }),
  });

  if (!response.ok) {
    if (response.status === 401) {
      useAuthStore.getState().logout();
    }
    const error = await response.json().catch(() => ({ detail: 'Failed to reject suggestions' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

/**
 * Bulk approve tag suggestions across multiple documents.
 */
export async function batchApproveSuggestions(
  suggestionIds: string[]
): Promise<ApprovalResponse> {
  const response = await fetch('/api/documents/tag-suggestions/approve-batch', {
    method: 'POST',
    headers: {
      ...getAuthHeaders(),
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ suggestion_ids: suggestionIds }),
  });

  if (!response.ok) {
    if (response.status === 401) {
      useAuthStore.getState().logout();
    }
    const error = await response.json().catch(() => ({ detail: 'Failed to batch approve suggestions' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}
