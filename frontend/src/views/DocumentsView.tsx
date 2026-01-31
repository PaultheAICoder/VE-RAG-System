import { useState, useEffect, useCallback, useRef } from 'react';
import { Plus } from 'lucide-react';
import { Button, Pagination, Alert } from '../components/ui';
import {
  DocumentTable,
  DocumentFilters,
  BulkActions,
  UploadModal,
  TagEditModal,
  ConfirmModal,
} from '../components/features/documents';
import {
  listDocuments,
  bulkDeleteDocuments,
  updateDocumentTags,
  bulkReprocessDocuments,
} from '../api/documents';
import { listTags } from '../api/tags';
import { useAuthStore } from '../stores/authStore';
import type { Document, Tag, DocumentStatus } from '../types';

const ITEMS_PER_PAGE = 20;
const AUTO_REFRESH_INTERVAL_ACTIVE = 3000; // 3 seconds when processing

export function DocumentsView() {
  const { user } = useAuthStore();
  const isAdmin = user?.role === 'admin' || user?.role === 'customer_admin';

  // Data state
  const [documents, setDocuments] = useState<Document[]>([]);
  const [tags, setTags] = useState<Tag[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Filter state
  const [search, setSearch] = useState('');
  const [selectedTagId, setSelectedTagId] = useState<string | null>(null);
  const [status, setStatus] = useState<DocumentStatus | null>(null);

  // Sort state
  const [sortBy, setSortBy] = useState('uploaded_at');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

  // Pagination state
  const [page, setPage] = useState(1);

  // Selection state
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());

  // Modal state
  const [showUpload, setShowUpload] = useState(false);
  const [showTagEdit, setShowTagEdit] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [actionLoading, setActionLoading] = useState(false);

  // Fetch tags on mount
  useEffect(() => {
    const fetchTags = async () => {
      try {
        const data = await listTags();
        setTags(data);
      } catch (err) {
        console.error('Failed to fetch tags:', err);
      }
    };
    fetchTags();
  }, []);

  // Fetch documents
  const fetchDocuments = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const offset = (page - 1) * ITEMS_PER_PAGE;
      const data = await listDocuments({
        limit: ITEMS_PER_PAGE,
        offset,
        search: search || undefined,
        tag_id: selectedTagId || undefined,
        status: status || undefined,
        sort_by: sortBy,
        sort_order: sortOrder,
      });

      setDocuments(data.documents);
      setTotal(data.total);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch documents');
    } finally {
      setLoading(false);
    }
  }, [page, search, selectedTagId, status, sortBy, sortOrder]);

  // Fetch documents when dependencies change
  useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments]);

  // Reset page when filters change
  useEffect(() => {
    setPage(1);
  }, [search, selectedTagId, status]);

  // Clear selection only when the set of document IDs changes (not just status updates)
  const documentIds = documents.map((d) => d.id).join(',');
  useEffect(() => {
    setSelectedIds(new Set());
  }, [documentIds]);

  // Check if any documents are processing or pending
  const isProcessingActive = documents.some(
    (doc) => doc.status === 'processing' || doc.status === 'pending'
  );

  // Auto-refresh when processing is active
  const refreshIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    // Clear existing interval
    if (refreshIntervalRef.current) {
      clearInterval(refreshIntervalRef.current);
      refreshIntervalRef.current = null;
    }

    // Set up auto-refresh only when processing is active
    if (isProcessingActive) {
      refreshIntervalRef.current = setInterval(() => {
        // Silent refresh - don't show loading state
        listDocuments({
          limit: ITEMS_PER_PAGE,
          offset: (page - 1) * ITEMS_PER_PAGE,
          search: search || undefined,
          tag_id: selectedTagId || undefined,
          status: status || undefined,
          sort_by: sortBy,
          sort_order: sortOrder,
        }).then((data) => {
          setDocuments(data.documents);
          setTotal(data.total);
        }).catch(() => {
          // Ignore errors during auto-refresh
        });
      }, AUTO_REFRESH_INTERVAL_ACTIVE);
    }

    return () => {
      if (refreshIntervalRef.current) {
        clearInterval(refreshIntervalRef.current);
      }
    };
  }, [isProcessingActive, page, search, selectedTagId, status, sortBy, sortOrder]);

  // Handle sort
  const handleSort = (field: string) => {
    if (sortBy === field) {
      setSortOrder((prev) => (prev === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortBy(field);
      setSortOrder('asc');
    }
  };

  // Handle tag click in table
  const handleTagClick = (tag: Tag) => {
    setSelectedTagId(tag.id);
  };

  // Handle bulk delete
  const handleBulkDelete = async () => {
    setActionLoading(true);
    try {
      const result = await bulkDeleteDocuments(Array.from(selectedIds));
      if (result.failed_count > 0) {
        setError(`Deleted ${result.deleted_count} documents, ${result.failed_count} failed`);
      }
      setShowDeleteConfirm(false);
      setSelectedIds(new Set());
      fetchDocuments();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete documents');
    } finally {
      setActionLoading(false);
    }
  };

  // Handle bulk tag edit
  const handleBulkTagEdit = async (tagIds: string[]) => {
    setActionLoading(true);
    try {
      // Update tags for each selected document
      const promises = Array.from(selectedIds).map((docId) =>
        updateDocumentTags(docId, tagIds)
      );
      await Promise.all(promises);
      setSelectedIds(new Set());
      fetchDocuments();
    } catch (err) {
      throw err; // Let TagEditModal handle the error
    } finally {
      setActionLoading(false);
    }
  };

  // Handle bulk reprocess - uses single API call, returns immediately
  const handleBulkReprocess = async () => {
    setActionLoading(true);
    try {
      const result = await bulkReprocessDocuments(Array.from(selectedIds));
      if (result.skipped > 0) {
        setError(`Queued ${result.queued} documents, ${result.skipped} skipped (already processing)`);
      }
      setSelectedIds(new Set());
      // Refresh to show documents changing to "pending" status
      fetchDocuments();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to reprocess documents');
    } finally {
      setActionLoading(false);
    }
  };

  const totalPages = Math.ceil(total / ITEMS_PER_PAGE);

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-heading font-bold text-gray-900 dark:text-white">
            Documents
          </h1>
          <p className="text-gray-500 dark:text-gray-400 mt-1">
            Manage your knowledge base documents
          </p>
        </div>

        {isAdmin && (
          <Button icon={Plus} onClick={() => setShowUpload(true)}>
            Upload
          </Button>
        )}
      </div>

      {/* Error Alert */}
      {error && (
        <Alert variant="danger" onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Filters */}
      <DocumentFilters
        search={search}
        onSearchChange={setSearch}
        selectedTagId={selectedTagId}
        onTagChange={setSelectedTagId}
        status={status}
        onStatusChange={setStatus}
        tags={tags}
      />

      {/* Bulk Actions (Admin only) */}
      {isAdmin && (
        <BulkActions
          selectedCount={selectedIds.size}
          onEditTags={() => setShowTagEdit(true)}
          onReprocess={handleBulkReprocess}
          onDelete={() => setShowDeleteConfirm(true)}
          disabled={actionLoading}
        />
      )}

      {/* Table */}
      <DocumentTable
        documents={documents}
        selectedIds={selectedIds}
        onSelectionChange={setSelectedIds}
        sortBy={sortBy}
        sortOrder={sortOrder}
        onSort={handleSort}
        onTagClick={handleTagClick}
        loading={loading}
      />

      {/* Pagination */}
      {totalPages > 1 && (
        <Pagination
          currentPage={page}
          totalPages={totalPages}
          totalItems={total}
          itemsPerPage={ITEMS_PER_PAGE}
          onPageChange={setPage}
        />
      )}

      {/* Upload Modal */}
      <UploadModal
        isOpen={showUpload}
        onClose={() => setShowUpload(false)}
        tags={tags}
        onUploadComplete={fetchDocuments}
      />

      {/* Tag Edit Modal */}
      <TagEditModal
        isOpen={showTagEdit}
        onClose={() => setShowTagEdit(false)}
        tags={tags}
        selectedCount={selectedIds.size}
        onSave={handleBulkTagEdit}
      />

      {/* Delete Confirmation Modal */}
      <ConfirmModal
        isOpen={showDeleteConfirm}
        onClose={() => setShowDeleteConfirm(false)}
        onConfirm={handleBulkDelete}
        title="Delete Documents"
        message={`Are you sure you want to delete ${selectedIds.size} document${selectedIds.size !== 1 ? 's' : ''}? This action cannot be undone.`}
        confirmLabel="Delete"
        isLoading={actionLoading}
        variant="danger"
      />
    </div>
  );
}
