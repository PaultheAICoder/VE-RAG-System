export type UserRole = 'admin' | 'customer_admin' | 'user';

export interface User {
  id: string;
  email: string;
  display_name: string;
  role: UserRole;
  is_active: boolean;
}

export interface NavItem {
  href: string;
  label: string;
  icon: React.ComponentType<{ size?: number }>;
  roles: UserRole[]; // Which roles can see this item
}

export interface Tag {
  id: string;
  name: string;
  display_name: string;
  description?: string | null;
  color?: string;
  owner_id?: string | null;
  is_system?: boolean;
}

export type DocumentStatus = 'pending' | 'processing' | 'ready' | 'failed';

export interface Document {
  id: string;
  original_filename: string;
  filename: string;
  file_type: string;
  file_size: number;
  status: DocumentStatus;
  title: string | null;
  description: string | null;
  chunk_count: number | null;
  page_count: number | null;
  word_count: number | null;
  processing_time_ms: number | null;
  error_message: string | null;
  tags: Tag[];
  uploaded_by: string;
  uploaded_at: string;
  processed_at: string | null;
}

export interface DocumentListResponse {
  documents: Document[];
  total: number;
  limit: number;
  offset: number;
}

export interface DocumentListParams {
  limit?: number;
  offset?: number;
  status?: DocumentStatus | null;
  tag_id?: string | null;
  search?: string | null;
  sort_by?: string;
  sort_order?: 'asc' | 'desc';
}

export interface BulkDeleteResult {
  id: string;
  status: 'deleted' | 'failed';
  error: string | null;
}

export interface BulkDeleteResponse {
  results: BulkDeleteResult[];
  deleted_count: number;
  failed_count: number;
}

// Chat types matching backend API schemas
export interface SourceInfo {
  source_id: string;
  document_id: string;
  chunk_index: number;
  title: string | null;
  snippet: string | null;
}

export interface ConfidenceInfo {
  overall: number; // 0-100
  retrieval: number;
  coverage: number;
  llm: number;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: SourceInfo[];
  confidence?: ConfidenceInfo;
  generation_time_ms?: number;
  was_routed?: boolean;
  routed_to?: string;
  route_reason?: string;
  created_at: string;
}

export interface ChatSession {
  id: string;
  title: string | null;
  created_at: string;
  updated_at: string;
  is_archived: boolean;
  message_count: number;
  last_message_preview?: string;
}

export interface SessionListResponse {
  sessions: ChatSession[];
  total: number;
  limit: number;
  offset: number;
}

export interface MessageListResponse {
  messages: ChatMessage[];
  has_more: boolean;
  total: number;
}

export interface SendMessageResponse {
  user_message: ChatMessage;
  assistant_message: ChatMessage;
  generation_time_ms: number;
  routing_decision?: string;
}

// User management types
export interface UserTag {
  id: string;
  name: string;
  display_name: string;
}

export interface UserWithTags extends User {
  must_reset_password: boolean;
  tags: UserTag[];
}

export interface UserCreate {
  email: string;
  display_name: string;
  password: string;
  role?: UserRole;
}

export interface UserUpdate {
  email?: string;
  display_name?: string;
  role?: UserRole;
  is_active?: boolean;
}

// Admin types
export interface ProcessingOptions {
  enable_ocr: boolean;
  force_full_page_ocr: boolean;
  ocr_language: string;
  table_extraction_mode: 'accurate' | 'fast';
  include_image_descriptions: boolean;
  query_routing_mode: 'retrieve_only' | 'retrieve_and_direct';
}

export interface ModelInfo {
  name: string;
  display_name: string;
  size_gb: number;
  parameters: string | null;
  quantization: string | null;
  recommended: boolean;
}

export interface ModelsResponse {
  available_models: ModelInfo[];
  embedding_models: ModelInfo[];
  chat_models: ModelInfo[];
  current_chat_model: string;
  current_embedding_model: string;
}

export interface TesseractStatus {
  available: boolean;
  version: string | null;
  languages: string[] | null;
}

export interface EasyOCRStatus {
  available: boolean;
  version: string | null;
}

export interface OCRStatus {
  tesseract: TesseractStatus;
  easyocr: EasyOCRStatus;
}

export interface DocumentParsingInfo {
  engine: string;
  version: string;
  type: string;
  capabilities: string[];
}

export interface EmbeddingsInfo {
  model: string;
  dimensions: number;
  vector_store: string;
  vector_store_url: string;
}

export interface ChatModelInfo {
  name: string;
  provider: string;
  capabilities: string[];
}

export interface InfrastructureStatus {
  ollama_url: string;
  ollama_status: 'healthy' | 'unhealthy';
  vector_db_status: 'healthy' | 'unhealthy';
}

export interface ArchitectureInfo {
  document_parsing: DocumentParsingInfo;
  embeddings: EmbeddingsInfo;
  chat_model: ChatModelInfo;
  infrastructure: InfrastructureStatus;
  ocr_status: OCRStatus;
  profile: string;
}

export interface FileStats {
  document_id: string;
  filename: string;
  chunk_count: number;
  status: string | null;
}

export interface KnowledgeBaseStats {
  total_chunks: number;
  unique_files: number;
  total_vectors: number;
  collection_name: string;
  files: FileStats[];
  storage_size_bytes: number | null;
  last_updated: string;
}

export interface HealthResponse {
  status: 'healthy' | 'unhealthy';
  version: string;
  database: string;
  rag_enabled: boolean;
  gradio_enabled: boolean;
  profile: string;
  backends: {
    vector: string;
    chunker: string;
    ocr_enabled: boolean;
  };
}

export interface ComponentHealth {
  name: string;
  status: 'healthy' | 'unhealthy' | 'degraded';
  version?: string;
  details?: Record<string, unknown>;
}

export interface RAGPipelineStatus {
  embedding_model: string;
  chat_model: string;
  chunker: string;
  stages: string[];
  all_stages_healthy: boolean;
}

export interface KnowledgeBaseSummary {
  total_documents: number;
  total_chunks: number;
  storage_size_mb?: number;
}

export interface ProcessingQueueStatus {
  pending: number;
  processing: number;
  failed: number;
  ready: number;
}

export interface DetailedHealthResponse {
  status: 'healthy' | 'unhealthy' | 'degraded';
  version: string;
  profile: string;
  api_server: ComponentHealth;
  ollama_llm: ComponentHealth;
  vector_db: ComponentHealth;
  rag_pipeline: RAGPipelineStatus;
  knowledge_base: KnowledgeBaseSummary;
  processing_queue: ProcessingQueueStatus;
  uptime_seconds: number;
  last_checked: string;
}

// RAG Tuning Settings
export interface RetrievalSettings {
  retrieval_top_k: number;
  retrieval_min_score: number;
  retrieval_enable_expansion: boolean;
}

export interface LLMSettings {
  llm_temperature: number;
  llm_max_response_tokens: number;
  llm_confidence_threshold: number;
}

export interface SettingsAuditEntry {
  id: string;
  setting_key: string;
  old_value: string | null;
  new_value: string;
  changed_by: string | null;
  changed_at: string;
  change_reason: string | null;
}

export interface SettingsAuditResponse {
  entries: SettingsAuditEntry[];
  total: number;
  limit: number;
  offset: number;
}

// Model Limits (for settings validation)
export interface ModelLimits {
  context_window: number;
  max_response: number;
  temperature_min: number;
  temperature_max: number;
}

export interface ModelLimitsResponse {
  current_model: string;
  limits: ModelLimits;
  all_models: Record<string, ModelLimits>;
}

// Advanced Settings (destructive, require reindex)
export interface AdvancedSettings {
  embedding_model: string;
  chunk_size: number;
  chunk_overlap: number;
  hnsw_ef_construct: number;
  hnsw_m: number;
  vector_backend: string;
}

// Reindex Job
export type ReindexJobStatus =
  | 'pending'
  | 'running'
  | 'paused'
  | 'completed'
  | 'failed'
  | 'aborted';

export interface ReindexJob {
  id: string;
  status: ReindexJobStatus;
  total_documents: number;
  processed_documents: number;
  failed_documents: number;
  progress_percent: number;
  current_document_id: string | null;
  error_message: string | null;
  started_at: string | null;
  completed_at: string | null;
  created_at: string;
  settings_changed: Record<string, unknown> | null;
  // Phase 3: Failure handling fields
  last_error: string | null;
  retry_count: number;
  max_retries: number;
  paused_at: string | null;
  paused_reason: string | null;
  auto_skip_failures: boolean;
}

export interface ReindexEstimate {
  total_documents: number;
  avg_processing_time_ms: number;
  estimated_total_seconds: number;
  estimated_time_str: string;
}

// Phase 3: Failure handling
export type ResumeAction = 'skip' | 'retry' | 'skip_all';

export interface ReindexFailureInfo {
  document_id: string;
  filename: string;
  status: string;
  error_message: string | null;
}

export interface ReindexFailuresResponse {
  job_id: string;
  failures: ReindexFailureInfo[];
  total_failures: number;
}

// Duplicate check types
export interface DuplicateInfo {
  filename: string;
  existing_id: string;
  existing_filename: string;
  uploaded_at: string; // ISO 8601
}

export interface CheckDuplicatesResponse {
  duplicates: DuplicateInfo[];
  unique: string[];
}

// Structured 409 error response for duplicates
export interface DuplicateErrorDetail {
  detail: string;
  error_code: string;
  existing_id: string;
  existing_filename: string;
  uploaded_at: string; // ISO 8601
}

// Standard error response structure
export interface UploadErrorResponse {
  detail: string;
  error_code?: string;
  existing_id?: string;
  existing_filename?: string;
  uploaded_at?: string;
}

// Upload results for modal
export type UploadResultStatus = 'success' | 'failed' | 'replaced' | 'skipped';

export interface UploadResult {
  filename: string;
  status: UploadResultStatus;
  error?: string;
  documentId?: string;
}

// =============================================================================
// Cache Settings Types (RAG Response Caching)
// =============================================================================

export interface CacheSettings {
  cache_enabled: boolean;
  cache_ttl_hours: number;
  cache_max_entries: number;
  cache_semantic_threshold: number;
  cache_min_confidence: number;
  cache_auto_warm_enabled: boolean;
  cache_auto_warm_count: number;
}

export interface CacheStats {
  enabled: boolean;
  total_entries: number;
  memory_entries: number;
  sqlite_entries: number;
  hit_count: number;
  miss_count: number;
  hit_rate: number;
  avg_response_time_cached_ms: number;
  avg_response_time_uncached_ms: number;
  storage_size_bytes: number;
  oldest_entry: string | null;
  newest_entry: string | null;
}

export interface HitRateTrendPoint {
  date: string;
  hit_rate: number;
  hits: number;
  misses: number;
}

export interface TopCachedQuery {
  query_text: string;
  hit_count: number;
  last_hit: string;
}

export interface ClearCacheResponse {
  cleared_entries: number;
  message: string;
}

export interface WarmCacheResponse {
  queued: number;
  message: string;
}

export interface WarmFileResponse {
  id: string;
  file_path: string;
  source_type: string;
  original_filename: string | null;
  total_queries: number;
  processed_queries: number;
  failed_queries: number;
  status: string;
  is_paused: boolean;
  is_cancel_requested: boolean;
  created_at: string | null;
  started_at: string | null;
  completed_at: string | null;
  created_by: string | null;
  error_message: string | null;
}

// =============================================================================
// Cache Warming Queue Types
// =============================================================================

export type WarmingJobStatus = 'pending' | 'running' | 'paused' | 'completed' | 'failed' | 'cancelled';

export interface WarmingJob {
  id: string;
  source_file: string | null;
  status: WarmingJobStatus;
  total: number;
  processed: number;
  success_count: number;
  failed_count: number;
  created_at: string | null;
  started_at: string | null;
  completed_at: string | null;
  triggered_by: string;
  // Transitional state flags
  is_cancel_requested?: boolean;
  is_paused?: boolean;
}

export interface WarmingJobListResponse {
  jobs: WarmingJob[];
  total_count: number;
}

// =============================================================================
// Chat Session Deletion Types
// =============================================================================

export interface DeleteSessionResponse {
  success: boolean;
  deleted_session_id: string;
  deleted_messages_count: number;
}

export interface BulkDeleteSessionsResponse {
  success: boolean;
  deleted_count: number;
  failed_ids: string[];
  total_messages_deleted: number;
}
