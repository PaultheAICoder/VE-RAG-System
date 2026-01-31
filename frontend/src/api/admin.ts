import { apiClient } from './client';
import type {
  ProcessingOptions,
  ModelsResponse,
  ArchitectureInfo,
  KnowledgeBaseStats,
  RetrievalSettings,
  LLMSettings,
  SettingsAuditResponse,
} from '../types';

/**
 * Get current processing options (system admin only).
 */
export async function getProcessingOptions(): Promise<ProcessingOptions> {
  return apiClient.get<ProcessingOptions>('/api/admin/processing-options');
}

/**
 * Update processing options (system admin only).
 */
export async function updateProcessingOptions(
  options: Partial<ProcessingOptions>
): Promise<ProcessingOptions> {
  return apiClient.patch<ProcessingOptions>('/api/admin/processing-options', options);
}

/**
 * Get available models and current selection (system admin only).
 */
export async function getModels(): Promise<ModelsResponse> {
  return apiClient.get<ModelsResponse>('/api/admin/models');
}

/**
 * Change the active chat model (system admin only).
 */
export async function changeChatModel(
  modelName: string
): Promise<{
  previous_model: string;
  current_model: string;
  success: boolean;
  message: string;
}> {
  return apiClient.patch<{
    previous_model: string;
    current_model: string;
    success: boolean;
    message: string;
  }>('/api/admin/models/chat', { model_name: modelName });
}

/**
 * Change the embedding model (system admin only).
 * Requires confirmation as it invalidates existing vectors.
 */
export async function changeEmbeddingModel(
  modelName: string,
  confirmReindex: boolean
): Promise<{
  previous_model: string;
  current_model: string;
  success: boolean;
  message: string;
  reindex_required: boolean;
  documents_affected: number;
}> {
  return apiClient.patch<{
    previous_model: string;
    current_model: string;
    success: boolean;
    message: string;
    reindex_required: boolean;
    documents_affected: number;
  }>('/api/admin/models/embedding', {
    model_name: modelName,
    confirm_reindex: confirmReindex,
  });
}

/**
 * Get system architecture information (system admin only).
 */
export async function getArchitectureInfo(): Promise<ArchitectureInfo> {
  return apiClient.get<ArchitectureInfo>('/api/admin/architecture');
}

/**
 * Get knowledge base statistics (system admin only).
 */
export async function getKnowledgeBaseStats(): Promise<KnowledgeBaseStats> {
  return apiClient.get<KnowledgeBaseStats>('/api/admin/knowledge-base/stats');
}

/**
 * Clear knowledge base (system admin only).
 * Deletes all vectors from the knowledge base.
 */
export async function clearKnowledgeBase(
  deleteSourceFiles: boolean = false
): Promise<{
  deleted_chunks: number;
  deleted_files: number;
  success: boolean;
  backend: string;
}> {
  return apiClient.delete<{
    deleted_chunks: number;
    deleted_files: number;
    success: boolean;
    backend: string;
  }>('/api/admin/knowledge-base', {
    confirm: true,
    delete_source_files: deleteSourceFiles,
  });
}

// =============================================================================
// RAG Tuning Settings
// =============================================================================

/**
 * Get current retrieval settings (system admin only).
 */
export async function getRetrievalSettings(): Promise<RetrievalSettings> {
  return apiClient.get<RetrievalSettings>('/api/admin/settings/retrieval');
}

/**
 * Update retrieval settings (system admin only).
 */
export async function updateRetrievalSettings(
  settings: Partial<RetrievalSettings>
): Promise<RetrievalSettings> {
  return apiClient.put<RetrievalSettings>('/api/admin/settings/retrieval', settings);
}

/**
 * Get current LLM response settings (system admin only).
 */
export async function getLLMSettings(): Promise<LLMSettings> {
  return apiClient.get<LLMSettings>('/api/admin/settings/llm');
}

/**
 * Update LLM response settings (system admin only).
 */
export async function updateLLMSettings(settings: Partial<LLMSettings>): Promise<LLMSettings> {
  return apiClient.put<LLMSettings>('/api/admin/settings/llm', settings);
}

/**
 * Get settings change audit history (system admin only).
 */
export async function getSettingsAudit(params?: {
  key?: string;
  limit?: number;
  offset?: number;
}): Promise<SettingsAuditResponse> {
  const searchParams = new URLSearchParams();
  if (params?.key) searchParams.set('key', params.key);
  if (params?.limit) searchParams.set('limit', params.limit.toString());
  if (params?.offset) searchParams.set('offset', params.offset.toString());

  const queryString = searchParams.toString();
  const url = queryString ? `/api/admin/settings/audit?${queryString}` : '/api/admin/settings/audit';
  return apiClient.get<SettingsAuditResponse>(url);
}
