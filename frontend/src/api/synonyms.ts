import { apiClient } from './client';
import type {
  Synonym,
  SynonymListResponse,
  SynonymListParams,
  SynonymCreate,
  SynonymUpdate,
} from '../types';

/**
 * List synonyms with pagination and filtering.
 */
export async function listSynonyms(params?: SynonymListParams): Promise<SynonymListResponse> {
  const searchParams = new URLSearchParams();
  if (params?.page) searchParams.set('page', params.page.toString());
  if (params?.page_size) searchParams.set('page_size', params.page_size.toString());
  if (params?.enabled !== undefined && params?.enabled !== null) {
    searchParams.set('enabled', params.enabled.toString());
  }
  if (params?.search) searchParams.set('search', params.search);

  const queryString = searchParams.toString();
  const url = queryString ? `/api/admin/synonyms?${queryString}` : '/api/admin/synonyms';
  return apiClient.get<SynonymListResponse>(url);
}

/**
 * Create a new synonym mapping.
 */
export async function createSynonym(data: SynonymCreate): Promise<Synonym> {
  return apiClient.post<Synonym>('/api/admin/synonyms', data);
}

/**
 * Update an existing synonym mapping.
 */
export async function updateSynonym(id: string, data: SynonymUpdate): Promise<Synonym> {
  return apiClient.put<Synonym>(`/api/admin/synonyms/${id}`, data);
}

/**
 * Delete a synonym mapping.
 */
export async function deleteSynonym(id: string): Promise<{ success: boolean; message: string }> {
  return apiClient.delete<{ success: boolean; message: string }>(`/api/admin/synonyms/${id}`);
}

/**
 * Toggle synonym enabled status.
 */
export async function toggleSynonymStatus(id: string, enabled: boolean): Promise<Synonym> {
  return apiClient.put<Synonym>(`/api/admin/synonyms/${id}`, { enabled });
}
