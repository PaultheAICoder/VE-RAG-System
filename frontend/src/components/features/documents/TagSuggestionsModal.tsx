import { useState, useEffect, useCallback } from 'react';
import { Check, X, CheckCheck, XCircle, Loader2 } from 'lucide-react';
import { Modal, Badge, Button } from '../../ui';
import { Checkbox } from '../../ui';
import { listTagSuggestions, approveSuggestions, rejectSuggestions } from '../../../api/suggestions';
import type { TagSuggestion } from '../../../types';

interface TagSuggestionsModalProps {
  isOpen: boolean;
  onClose: () => void;
  documentId: string | null;
  documentName: string;
  onActionComplete: () => void;
}

export function TagSuggestionsModal({
  isOpen,
  onClose,
  documentId,
  documentName,
  onActionComplete,
}: TagSuggestionsModalProps) {
  const [suggestions, setSuggestions] = useState<TagSuggestion[]>([]);
  const [loading, setLoading] = useState(false);
  const [actionLoading, setActionLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());

  const fetchSuggestions = useCallback(async () => {
    if (!documentId) return;
    setLoading(true);
    setError(null);
    try {
      const data = await listTagSuggestions(documentId, 'pending');
      setSuggestions(data.suggestions);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load suggestions');
    } finally {
      setLoading(false);
    }
  }, [documentId]);

  useEffect(() => {
    if (isOpen && documentId) {
      fetchSuggestions();
      setSelectedIds(new Set());
    }
  }, [isOpen, documentId, fetchSuggestions]);

  const handleApprove = async (ids: string[]) => {
    if (!documentId || ids.length === 0) return;
    setActionLoading(true);
    setError(null);
    try {
      const result = await approveSuggestions(documentId, ids);
      if (result.failed_count > 0) {
        const failures = result.results
          .filter((r) => r.status === 'failed')
          .map((r) => r.error)
          .join('; ');
        setError(`${result.processed_count} approved, ${result.failed_count} failed: ${failures}`);
      }
      setSelectedIds(new Set());
      await fetchSuggestions();
      onActionComplete();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to approve suggestions');
    } finally {
      setActionLoading(false);
    }
  };

  const handleReject = async (ids: string[]) => {
    if (!documentId || ids.length === 0) return;
    setActionLoading(true);
    setError(null);
    try {
      const result = await rejectSuggestions(documentId, ids);
      if (result.failed_count > 0) {
        const failures = result.results
          .filter((r) => r.status === 'failed')
          .map((r) => r.error)
          .join('; ');
        setError(`${result.processed_count} rejected, ${result.failed_count} failed: ${failures}`);
      }
      setSelectedIds(new Set());
      await fetchSuggestions();
      onActionComplete();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to reject suggestions');
    } finally {
      setActionLoading(false);
    }
  };

  const handleToggleSelect = (id: string) => {
    const next = new Set(selectedIds);
    if (next.has(id)) {
      next.delete(id);
    } else {
      next.add(id);
    }
    setSelectedIds(next);
  };

  const handleSelectAll = () => {
    if (selectedIds.size === suggestions.length) {
      setSelectedIds(new Set());
    } else {
      setSelectedIds(new Set(suggestions.map((s) => s.id)));
    }
  };

  const allSelected = suggestions.length > 0 && selectedIds.size === suggestions.length;
  const someSelected = selectedIds.size > 0 && !allSelected;

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title={`Tag Suggestions - ${documentName}`}
      size="lg"
    >
      <div className="space-y-4">
        {/* Error message */}
        {error && (
          <div className="p-3 bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-400 text-sm rounded-lg">
            {error}
          </div>
        )}

        {/* Loading state */}
        {loading && (
          <div className="flex items-center justify-center py-8 text-gray-500">
            <Loader2 className="animate-spin mr-2" size={20} />
            Loading suggestions...
          </div>
        )}

        {/* Empty state */}
        {!loading && suggestions.length === 0 && (
          <div className="py-8 text-center text-gray-500 dark:text-gray-400">
            No pending tag suggestions for this document.
          </div>
        )}

        {/* Suggestions list */}
        {!loading && suggestions.length > 0 && (
          <>
            {/* Bulk actions bar */}
            <div className="flex items-center justify-between pb-2 border-b border-gray-200 dark:border-gray-700">
              <div className="flex items-center gap-3">
                <Checkbox
                  checked={allSelected}
                  indeterminate={someSelected}
                  onChange={handleSelectAll}
                />
                <span className="text-sm text-gray-600 dark:text-gray-400">
                  {selectedIds.size > 0
                    ? `${selectedIds.size} selected`
                    : `${suggestions.length} suggestion${suggestions.length !== 1 ? 's' : ''}`}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <Button
                  variant="primary"
                  size="sm"
                  icon={CheckCheck}
                  onClick={() => handleApprove(suggestions.map((s) => s.id))}
                  disabled={actionLoading}
                >
                  Approve All
                </Button>
                <Button
                  variant="danger"
                  size="sm"
                  icon={XCircle}
                  onClick={() => handleReject(suggestions.map((s) => s.id))}
                  disabled={actionLoading}
                >
                  Reject All
                </Button>
              </div>
            </div>

            {/* Suggestion rows */}
            <div className="max-h-80 overflow-y-auto space-y-2">
              {suggestions.map((suggestion) => (
                <div
                  key={suggestion.id}
                  className={`flex items-center justify-between p-3 rounded-lg border transition-colors ${
                    selectedIds.has(suggestion.id)
                      ? 'border-primary/30 bg-primary/5'
                      : 'border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700/30'
                  }`}
                >
                  <div className="flex items-center gap-3 min-w-0 flex-1">
                    <Checkbox
                      checked={selectedIds.has(suggestion.id)}
                      onChange={() => handleToggleSelect(suggestion.id)}
                    />
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className="font-medium text-gray-900 dark:text-white text-sm">
                          {suggestion.display_name}
                        </span>
                        <span className="text-xs text-gray-500 dark:text-gray-400">
                          {suggestion.namespace}
                        </span>
                        <Badge variant={suggestion.source === 'path' ? 'primary' : 'default'}>
                          {suggestion.source}
                        </Badge>
                        <span className="text-xs text-gray-500 dark:text-gray-400">
                          {Math.round(suggestion.confidence * 100)}%
                        </span>
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-1 ml-2 shrink-0">
                    <button
                      onClick={() => handleApprove([suggestion.id])}
                      disabled={actionLoading}
                      className="p-1.5 text-green-600 hover:bg-green-100 dark:hover:bg-green-900/30 rounded-lg transition-colors disabled:opacity-50"
                      title="Approve"
                    >
                      <Check size={16} />
                    </button>
                    <button
                      onClick={() => handleReject([suggestion.id])}
                      disabled={actionLoading}
                      className="p-1.5 text-red-600 hover:bg-red-100 dark:hover:bg-red-900/30 rounded-lg transition-colors disabled:opacity-50"
                      title="Reject"
                    >
                      <X size={16} />
                    </button>
                  </div>
                </div>
              ))}
            </div>

            {/* Footer with selected actions */}
            {selectedIds.size > 0 && (
              <div className="flex items-center justify-between pt-2 border-t border-gray-200 dark:border-gray-700">
                <span className="text-sm text-gray-600 dark:text-gray-400">
                  {selectedIds.size} selected
                </span>
                <div className="flex items-center gap-2">
                  <Button
                    variant="primary"
                    size="sm"
                    icon={Check}
                    onClick={() => handleApprove(Array.from(selectedIds))}
                    disabled={actionLoading}
                  >
                    Approve Selected
                  </Button>
                  <Button
                    variant="danger"
                    size="sm"
                    icon={X}
                    onClick={() => handleReject(Array.from(selectedIds))}
                    disabled={actionLoading}
                  >
                    Reject Selected
                  </Button>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </Modal>
  );
}
