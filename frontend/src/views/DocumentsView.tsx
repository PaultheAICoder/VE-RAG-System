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
  TagSuggestionsModal,
  NamespaceFilters,
} from '../components/features/documents';
import {
  listDocuments,
  bulkDeleteDocuments,
  updateDocumentTags,
  bulkReprocessDocuments,
} from '../api/documents';
import { listTagSuggestions } from '../api/suggestions';
import { listTags, getTagFacets } from '../api/tags';
import { getHealth } from '../api/health';
import { useAuthStore } from '../stores/authStore';
import { useDocumentsStore } from '../stores/documentsStore';
import type { Document, Tag, TagFacetItem } from '../types';

const ITEMS_PER_PAGE = 20;
const AUTO_REFRESH_INTERVAL_ACTIVE = 3000; // 3 seconds when processing

export function DocumentsView() {
  const { user } = useAuthStore();
  const isAdmin = user?.role === 'admin' || user?.role === 'customer_admin';

  // Get persisted state from store
  const {
    search,
    setSearch,
    selectedTagId,
    setTagFilter,
    status,
    setStatusFilter,
    namespaceFilter,
    setNamespaceFilter,
    sortBy,
    sortOrder,
    setSort,
    page,
    setPage,
    validateTagFilter,
    clampPage,
  } = useDocumentsStore();

  // Data state (not persisted - fetched fresh)
  const [documents, setDocuments] = useState<Document[]>([]);
  const [tags, setTags] = useState<Tag[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Facets state (not persisted - fetched fresh)
  const [facets, setFacets] = useState<Record<string, TagFacetItem[]>>({});
  const [facetsLoading, setFacetsLoading] = useState(false);

  // Selection state (not persisted - transient per visit)
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());

  // Modal state
  const [showUpload, setShowUpload] = useState(false);
  const [showTagEdit, setShowTagEdit] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [selectedDocForSuggestions, setSelectedDocForSuggestions] = useState<Document | null>(null);
  const [suggestionCounts, setSuggestionCounts] = useState<Map<string, number>>(new Map());
  const [actionLoading, setActionLoading] = useState(false);
  const [autoTaggingEnabled, setAutoTaggingEnabled] = useState(false);

  // Fetch auto-tagging status on mount
  useEffect(() => {
    const fetchAutoTagStatus = async () => {
      try {
        const health = await getHealth();
        setAutoTaggingEnabled(health.auto_tagging?.enabled ?? false);
      } catch {
        // Default to false if health check fails
      }
    };
    fetchAutoTagStatus();
  }, []);

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

  // Fetch tag facets on mount
  useEffect(() => {
    const fetchFacets = async () => {
      setFacetsLoading(true);
      try {
        const data = await getTagFacets();
        setFacets(data);
      } catch (err) {
        console.error('Failed to fetch tag facets:', err);
      } finally {
        setFacetsLoading(false);
      }
    };
    fetchFacets();
  }, []);

  // Validate persisted tag filter when tags load
  useEffect(() => {
    if (tags.length > 0) {
      validateTagFilter(tags.map((t) => t.id));
    }
  }, [tags, validateTagFilter]);

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
        tag_namespace: namespaceFilter?.namespace || undefined,
        tag_value: namespaceFilter?.value || undefined,
      });

      setDocuments(data.documents);
      setTotal(data.total);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch documents');
    } finally {
      setLoading(false);
    }
  }, [page, search, selectedTagId, status, sortBy, sortOrder, namespaceFilter]);

  // Fetch suggestion counts for visible documents (admin only)
  const fetchSuggestionCounts = useCallback(async (docs: Document[]) => {
    if (!isAdmin || docs.length === 0) {
      setSuggestionCounts(new Map());
      return;
    }
    try {
      const results = await Promise.all(
        docs.map(async (doc) => {
          try {
            const data = await listTagSuggestions(doc.id, 'pending');
            return [doc.id, data.total] as [string, number];
          } catch {
            return [doc.id, 0] as [string, number];
          }
        })
      );
      setSuggestionCounts(new Map(results));
    } catch {
      setSuggestionCounts(new Map());
    }
  }, [isAdmin]);

  // Fetch documents when dependencies change
  useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments]);

  // Clamp page to valid range when total changes
  useEffect(() => {
    if (total > 0) {
      clampPage(total);
    }
  }, [total, clampPage]);

  // Fetch suggestion counts when documents change
  useEffect(() => {
    if (documents.length > 0) {
      fetchSuggestionCounts(documents);
    }
  }, [documents, fetchSuggestionCounts]);

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
          tag_namespace: namespaceFilter?.namespace || undefined,
          tag_value: namespaceFilter?.value || undefined,
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
  }, [isProcessingActive, page, search, selectedTagId, status, sortBy, sortOrder, namespaceFilter]);

  // Handle sort
  const handleSort = (field: string) => {
    if (sortBy === field) {
      setSort(field, sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSort(field, 'asc');
    }
  };

  // Handle tag click in table
  const handleTagClick = (tag: Tag) => {
    setTagFilter(tag.id);
  };

  // Handle suggestion badge click
  const handleSuggestionClick = (doc: Document) => {
    setSelectedDocForSuggestions(doc);
    setShowSuggestions(true);
  };

  // Handle suggestion action complete (refresh documents and counts)
  const handleSuggestionActionComplete = () => {
    fetchDocuments();
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
        onTagChange={setTagFilter}
        status={status}
        onStatusChange={setStatusFilter}
        tags={tags}
      />

      {/* Namespace Filters */}
      <NamespaceFilters
        facets={facets}
        activeNamespace={namespaceFilter?.namespace ?? null}
        activeValue={namespaceFilter?.value ?? null}
        onFilterChange={setNamespaceFilter}
        loading={facetsLoading}
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
        suggestionCounts={isAdmin ? suggestionCounts : undefined}
        onSuggestionClick={isAdmin ? handleSuggestionClick : undefined}
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
        autoTaggingEnabled={autoTaggingEnabled}
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

      {/* Tag Suggestions Modal */}
      <TagSuggestionsModal
        isOpen={showSuggestions}
        onClose={() => {
          setShowSuggestions(false);
          setSelectedDocForSuggestions(null);
        }}
        documentId={selectedDocForSuggestions?.id ?? null}
        documentName={selectedDocForSuggestions?.original_filename ?? ''}
        onActionComplete={handleSuggestionActionComplete}
      />
    </div>
  );
}
