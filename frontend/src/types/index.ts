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
}

export interface Document {
  id: string;
  filename: string;
  status: 'pending' | 'processing' | 'ready' | 'failed';
  tags: Tag[];
  uploaded_at: string;
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
