import { apiClient } from './client';
import type {
  ProcessingOptions,
  ModelsResponse,
  ArchitectureInfo,
  KnowledgeBaseStats,
  RetrievalSettings,
  LLMSettings,
  SettingsAuditResponse,
  ModelLimitsResponse,
  AdvancedSettings,
  ReindexJob,
  ReindexEstimate,
  ResumeAction,
  ReindexFailuresResponse,
  StrategyListResponse,
  StrategyDetailResponse,
  ActiveStrategyResponse,
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
 * Get model-specific limits for settings validation (system admin only).
 * Returns current model limits and all known model limits.
 */
export async function getModelLimits(): Promise<ModelLimitsResponse> {
  return apiClient.get<ModelLimitsResponse>('/api/admin/model-limits');
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

// =============================================================================
// Advanced Settings & Reindex
// =============================================================================

/**
 * Get advanced RAG settings (system admin only).
 * These settings require a reindex when changed.
 */
export async function getAdvancedSettings(): Promise<AdvancedSettings> {
  return apiClient.get<AdvancedSettings>('/api/admin/settings/advanced');
}

/**
 * Update advanced RAG settings (system admin only).
 * Requires confirmReindex=true to proceed.
 */
export async function updateAdvancedSettings(
  settings: Partial<AdvancedSettings>,
  confirmReindex: boolean
): Promise<AdvancedSettings> {
  return apiClient.put<AdvancedSettings>('/api/admin/settings/advanced', {
    ...settings,
    confirm_reindex: confirmReindex,
  });
}

/**
 * Get time estimate for reindex operation (system admin only).
 */
export async function getReindexEstimate(): Promise<ReindexEstimate> {
  return apiClient.get<ReindexEstimate>('/api/admin/reindex/estimate');
}

/**
 * Start a background reindex operation (system admin only).
 * Returns the created job. Poll getReindexStatus for progress.
 */
export async function startReindex(): Promise<ReindexJob> {
  return apiClient.post<ReindexJob>('/api/admin/reindex/start', { confirm: true });
}

/**
 * Get current reindex job status (system admin only).
 * Returns null if no job is running.
 */
export async function getReindexStatus(): Promise<ReindexJob | null> {
  return apiClient.get<ReindexJob | null>('/api/admin/reindex/status');
}

/**
 * Abort current reindex operation (system admin only).
 */
export async function abortReindex(): Promise<ReindexJob> {
  return apiClient.post<ReindexJob>('/api/admin/reindex/abort', {});
}

/**
 * Get reindex job history (system admin only).
 */
export async function getReindexHistory(limit?: number): Promise<ReindexJob[]> {
  const url = limit ? `/api/admin/reindex/history?limit=${limit}` : '/api/admin/reindex/history';
  return apiClient.get<ReindexJob[]>(url);
}

// =============================================================================
// Reindex Failure Handling (Phase 3)
// =============================================================================

/**
 * Pause a running reindex job (system admin only).
 * Use this to manually pause a reindex operation.
 */
export async function pauseReindex(): Promise<ReindexJob> {
  return apiClient.post<ReindexJob>('/api/admin/reindex/pause', {});
}

/**
 * Resume a paused reindex job (system admin only).
 * @param action - What to do: 'skip' (skip failed doc), 'retry' (retry failed doc), or 'skip_all' (auto-skip all failures)
 */
export async function resumeReindex(action: ResumeAction): Promise<ReindexJob> {
  return apiClient.post<ReindexJob>('/api/admin/reindex/resume', { action });
}

/**
 * Get details about failed documents in a reindex job (system admin only).
 * @param jobId - Optional job ID. If not provided, returns failures for active job.
 */
export async function getReindexFailures(jobId?: string): Promise<ReindexFailuresResponse> {
  const url = jobId
    ? `/api/admin/reindex/failures?job_id=${jobId}`
    : '/api/admin/reindex/failures';
  return apiClient.get<ReindexFailuresResponse>(url);
}

/**
 * Retry a specific failed document in a reindex job (system admin only).
 * @param documentId - ID of the document to retry
 * @param jobId - Optional job ID. If not provided, uses active job.
 */
export async function retryReindexDocument(
  documentId: string,
  jobId?: string
): Promise<ReindexJob> {
  const url = jobId
    ? `/api/admin/reindex/retry/${documentId}?job_id=${jobId}`
    : `/api/admin/reindex/retry/${documentId}`;
  return apiClient.post<ReindexJob>(url, {});
}

// =============================================================================
// Auto-Tagging Strategy Management
// =============================================================================

/**
 * Get all available auto-tagging strategies (system admin only).
 */
export async function getAutoTagStrategies(): Promise<StrategyListResponse> {
  return apiClient.get<StrategyListResponse>('/api/admin/auto-tagging/strategies');
}

/**
 * Get detailed info for a specific strategy (system admin only).
 */
export async function getAutoTagStrategy(strategyId: string): Promise<StrategyDetailResponse> {
  return apiClient.get<StrategyDetailResponse>(
    `/api/admin/auto-tagging/strategies/${strategyId}`
  );
}

/**
 * Get the currently active auto-tagging strategy (system admin only).
 */
export async function getActiveAutoTagStrategy(): Promise<ActiveStrategyResponse> {
  return apiClient.get<ActiveStrategyResponse>('/api/admin/auto-tagging/active');
}

/**
 * Switch the active auto-tagging strategy (system admin only).
 */
export async function switchActiveAutoTagStrategy(
  strategyId: string
): Promise<ActiveStrategyResponse> {
  return apiClient.put<ActiveStrategyResponse>('/api/admin/auto-tagging/active', {
    strategy_id: strategyId,
  });
}
