import { apiClient } from './client';
import type { ReviewQueueListResponse, ReviewItemResolveResponse } from '../types';

/**
 * List review queue items (admin only).
 * @param status - Optional filter: 'pending' | 'accepted' | 'corrected' | 'dismissed'
 * @param page - Page number (1-based)
 * @param pageSize - Items per page
 */
export async function listReviewQueue(params?: {
  status?: string;
  page?: number;
  pageSize?: number;
}): Promise<ReviewQueueListResponse> {
  const searchParams = new URLSearchParams();
  if (params?.status) searchParams.set('status', params.status);
  if (params?.page) searchParams.set('page', params.page.toString());
  if (params?.pageSize) searchParams.set('page_size', params.pageSize.toString());

  const queryString = searchParams.toString();
  const url = queryString
    ? `/api/admin/review-queue?${queryString}`
    : '/api/admin/review-queue';
  return apiClient.get<ReviewQueueListResponse>(url);
}

/**
 * Approve a review queue item (admin only).
 * Sets review_status to 'accepted'.
 */
export async function approveReviewItem(itemId: string): Promise<ReviewItemResolveResponse> {
  return apiClient.post<ReviewItemResolveResponse>(
    `/api/admin/review-queue/${itemId}/approve`
  );
}

/**
 * Reject a review queue item (admin only).
 * Sets review_status to 'dismissed'.
 */
export async function rejectReviewItem(itemId: string): Promise<ReviewItemResolveResponse> {
  return apiClient.post<ReviewItemResolveResponse>(
    `/api/admin/review-queue/${itemId}/reject`
  );
}
