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
