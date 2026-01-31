import { useAuthStore } from '../stores/authStore';
import type {
  Document,
  DocumentListResponse,
  DocumentListParams,
  BulkDeleteResponse,
} from '../types';

const getAuthHeaders = (): HeadersInit => {
  const token = useAuthStore.getState().token;
  const headers: HeadersInit = {};
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  return headers;
};

/**
 * List documents with filtering and pagination.
 */
export async function listDocuments(
  params: DocumentListParams = {}
): Promise<DocumentListResponse> {
  const searchParams = new URLSearchParams();

  if (params.limit) searchParams.set('limit', params.limit.toString());
  if (params.offset) searchParams.set('offset', params.offset.toString());
  if (params.status) searchParams.set('status', params.status);
  if (params.tag_id) searchParams.set('tag_id', params.tag_id);
  if (params.search) searchParams.set('search', params.search);
  if (params.sort_by) searchParams.set('sort_by', params.sort_by);
  if (params.sort_order) searchParams.set('sort_order', params.sort_order);

  const response = await fetch(`/api/documents?${searchParams}`, {
    headers: getAuthHeaders(),
  });

  if (!response.ok) {
    if (response.status === 401) {
      useAuthStore.getState().logout();
    }
    const error = await response.json().catch(() => ({ detail: 'Failed to fetch documents' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

/**
 * Get a single document by ID.
 */
export async function getDocument(id: string): Promise<Document> {
  const response = await fetch(`/api/documents/${id}`, {
    headers: getAuthHeaders(),
  });

  if (!response.ok) {
    if (response.status === 401) {
      useAuthStore.getState().logout();
    }
    const error = await response.json().catch(() => ({ detail: 'Failed to fetch document' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

/**
 * Upload a document with tags and optional progress tracking.
 */
export async function uploadDocument(
  file: File,
  tagIds: string[],
  onProgress?: (percent: number) => void
): Promise<Document> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const formData = new FormData();

    formData.append('file', file);
    // FastAPI expects repeated form fields for list
    tagIds.forEach((tagId) => {
      formData.append('tag_ids', tagId);
    });

    xhr.upload.addEventListener('progress', (e) => {
      if (e.lengthComputable && onProgress) {
        const percent = Math.round((e.loaded / e.total) * 100);
        onProgress(percent);
      }
    });

    xhr.addEventListener('load', () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const data = JSON.parse(xhr.responseText);
          resolve(data);
        } catch {
          reject(new Error('Invalid response'));
        }
      } else if (xhr.status === 401) {
        useAuthStore.getState().logout();
        reject(new Error('Unauthorized'));
      } else {
        try {
          const error = JSON.parse(xhr.responseText);
          reject(new Error(error.detail || `Upload failed: ${xhr.status}`));
        } catch {
          reject(new Error(`Upload failed: ${xhr.status}`));
        }
      }
    });

    xhr.addEventListener('error', () => {
      reject(new Error('Network error'));
    });

    xhr.addEventListener('abort', () => {
      reject(new Error('Upload cancelled'));
    });

    xhr.open('POST', '/api/documents/upload');

    const token = useAuthStore.getState().token;
    if (token) {
      xhr.setRequestHeader('Authorization', `Bearer ${token}`);
    }

    xhr.send(formData);
  });
}

/**
 * Delete a single document.
 */
export async function deleteDocument(id: string): Promise<void> {
  const response = await fetch(`/api/documents/${id}`, {
    method: 'DELETE',
    headers: getAuthHeaders(),
  });

  if (!response.ok) {
    if (response.status === 401) {
      useAuthStore.getState().logout();
    }
    const error = await response.json().catch(() => ({ detail: 'Failed to delete document' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }
}

/**
 * Bulk delete multiple documents.
 */
export async function bulkDeleteDocuments(
  documentIds: string[]
): Promise<BulkDeleteResponse> {
  const response = await fetch('/api/documents/bulk-delete', {
    method: 'POST',
    headers: {
      ...getAuthHeaders(),
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ document_ids: documentIds }),
  });

  if (!response.ok) {
    if (response.status === 401) {
      useAuthStore.getState().logout();
    }
    const error = await response.json().catch(() => ({ detail: 'Failed to delete documents' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

/**
 * Update document tags.
 */
export async function updateDocumentTags(
  documentId: string,
  tagIds: string[]
): Promise<Document> {
  const response = await fetch(`/api/documents/${documentId}/tags`, {
    method: 'PATCH',
    headers: {
      ...getAuthHeaders(),
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ tag_ids: tagIds }),
  });

  if (!response.ok) {
    if (response.status === 401) {
      useAuthStore.getState().logout();
    }
    const error = await response.json().catch(() => ({ detail: 'Failed to update tags' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

/**
 * Reprocess a document.
 */
export async function reprocessDocument(documentId: string): Promise<Document> {
  const response = await fetch(`/api/documents/${documentId}/reprocess`, {
    method: 'POST',
    headers: {
      ...getAuthHeaders(),
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    if (response.status === 401) {
      useAuthStore.getState().logout();
    }
    const error = await response.json().catch(() => ({ detail: 'Failed to reprocess document' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}
