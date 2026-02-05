import { apiClient } from './client';
import type {
  CuratedQA,
  CuratedQACreate,
  CuratedQAUpdate,
  CuratedQAListResponse,
  CuratedQAListParams,
} from '../types';

/**
 * List curated Q&A pairs with pagination and filtering.
 */
export async function listCuratedQA(params?: CuratedQAListParams): Promise<CuratedQAListResponse> {
  const searchParams = new URLSearchParams();
  if (params?.page) searchParams.set('page', params.page.toString());
  if (params?.page_size) searchParams.set('page_size', params.page_size.toString());
  if (params?.enabled !== undefined && params?.enabled !== null) {
    searchParams.set('enabled', params.enabled.toString());
  }
  if (params?.search) searchParams.set('search', params.search);

  const queryString = searchParams.toString();
  const url = queryString ? `/api/admin/qa?${queryString}` : '/api/admin/qa';
  return apiClient.get<CuratedQAListResponse>(url);
}

/**
 * Create a new curated Q&A pair.
 */
export async function createCuratedQA(data: CuratedQACreate): Promise<CuratedQA> {
  return apiClient.post<CuratedQA>('/api/admin/qa', data);
}

/**
 * Update a curated Q&A pair.
 */
export async function updateCuratedQA(id: string, data: CuratedQAUpdate): Promise<CuratedQA> {
  return apiClient.put<CuratedQA>(`/api/admin/qa/${id}`, data);
}

/**
 * Delete a curated Q&A pair.
 */
export async function deleteCuratedQA(id: string): Promise<{ success: boolean; message: string }> {
  return apiClient.delete<{ success: boolean; message: string }>(`/api/admin/qa/${id}`);
}
