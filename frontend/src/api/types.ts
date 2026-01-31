// Auth responses (from ai_ready_rag/api/auth.py)
export interface LoginResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
  user: UserResponse;
  setup_required: boolean;
}

export interface UserResponse {
  id: string;
  email: string;
  display_name: string;
  role: 'admin' | 'customer_admin' | 'user';
  is_active: boolean;
}

// Error response
export interface APIError {
  detail: string;
}
