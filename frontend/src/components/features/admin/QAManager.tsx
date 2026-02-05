import { useState, useEffect, useCallback } from 'react';
import { Plus, Search, RefreshCw, Pencil, Trash2, ToggleLeft, ToggleRight, Eye } from 'lucide-react';
import { Button, Alert, Input, Select, Pagination, Card, Badge } from '../../ui';
import { QAForm } from './QAForm';
import { ConfirmModal } from './ConfirmModal';
import {
  listCuratedQA,
  createCuratedQA,
  updateCuratedQA,
  deleteCuratedQA,
} from '../../../api/qa';
import { stripHtml, truncateText } from '../../../utils/sanitize';
import type {
  CuratedQA,
  CuratedQACreate,
  CuratedQAUpdate,
  CuratedQAListParams,
} from '../../../types';

const ENABLED_OPTIONS = [
  { value: '', label: 'All Status' },
  { value: 'true', label: 'Enabled' },
  { value: 'false', label: 'Disabled' },
];

const PAGE_SIZE = 9;

interface QACardProps {
  qa: CuratedQA;
  onEdit: (qa: CuratedQA) => void;
  onDelete: (qa: CuratedQA) => void;
  onToggleEnabled: (qa: CuratedQA) => void;
}

function QACard({ qa, onEdit, onDelete, onToggleEnabled }: QACardProps) {
  const answerPreview = truncateText(stripHtml(qa.answer), 150);
  const formattedDate = new Date(qa.created_at).toLocaleDateString();

  return (
    <Card className="hover:shadow-md transition-shadow relative">
      {!qa.enabled && (
        <div className="absolute top-2 right-2">
          <Badge variant="default">Disabled</Badge>
        </div>
      )}

      <div className="flex items-start justify-between mb-3">
        <div className="flex flex-wrap gap-1.5 pr-16">
          {qa.keywords.slice(0, 3).map((keyword, index) => (
            <Badge key={index} variant="primary">
              {keyword}
            </Badge>
          ))}
          {qa.keywords.length > 3 && (
            <Badge variant="default">+{qa.keywords.length - 3}</Badge>
          )}
        </div>

        <div className="flex items-center gap-1 flex-shrink-0 absolute top-4 right-4">
          <button
            onClick={() => onToggleEnabled(qa)}
            className={`
              p-1.5 rounded transition-colors
              ${qa.enabled
                ? 'text-green-500 hover:bg-green-50 dark:hover:bg-green-900/20'
                : 'text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700'
              }
            `}
            title={qa.enabled ? 'Disable' : 'Enable'}
          >
            {qa.enabled ? <ToggleRight size={18} /> : <ToggleLeft size={18} />}
          </button>
          <button
            onClick={() => onEdit(qa)}
            className="p-1.5 text-gray-400 hover:text-primary hover:bg-primary/10 rounded transition-colors"
            title="Edit"
          >
            <Pencil size={16} />
          </button>
          <button
            onClick={() => onDelete(qa)}
            className="p-1.5 text-gray-400 hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20 rounded transition-colors"
            title="Delete"
          >
            <Trash2 size={16} />
          </button>
        </div>
      </div>

      <div className="text-sm text-gray-500 dark:text-gray-400 mb-2">
        <span className="font-medium">Source:</span> {qa.source_reference}
      </div>

      <p className="text-gray-700 dark:text-gray-300 text-sm mb-3 line-clamp-3">
        {answerPreview}
      </p>

      <div className="flex items-center justify-between pt-3 border-t border-gray-100 dark:border-gray-700 text-xs text-gray-500 dark:text-gray-400">
        <div className="flex items-center gap-4">
          <span>
            <span className="font-medium">Confidence:</span> {qa.confidence}%
          </span>
          <span>
            <span className="font-medium">Priority:</span> {qa.priority}
          </span>
          <span className="flex items-center gap-1">
            <Eye size={12} />
            {qa.access_count}
          </span>
        </div>
        <span>{formattedDate}</span>
      </div>
    </Card>
  );
}

export function QAManager() {
  const [qaList, setQaList] = useState<CuratedQA[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const [search, setSearch] = useState('');
  const [enabledFilter, setEnabledFilter] = useState('');
  const [page, setPage] = useState(1);

  const [showForm, setShowForm] = useState(false);
  const [editingQA, setEditingQA] = useState<CuratedQA | null>(null);
  const [deletingQA, setDeletingQA] = useState<CuratedQA | null>(null);
  const [actionLoading, setActionLoading] = useState(false);

  const [debouncedSearch, setDebouncedSearch] = useState('');

  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedSearch(search);
      setPage(1);
    }, 300);
    return () => clearTimeout(timer);
  }, [search]);

  const fetchQA = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const params: CuratedQAListParams = {
        page,
        page_size: PAGE_SIZE,
      };

      if (debouncedSearch) {
        params.search = debouncedSearch;
      }

      if (enabledFilter) {
        params.enabled = enabledFilter === 'true';
      }

      const data = await listCuratedQA(params);
      setQaList(data.qa_pairs);
      setTotal(data.total);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load Q&A pairs');
    } finally {
      setLoading(false);
    }
  }, [page, debouncedSearch, enabledFilter]);

  useEffect(() => {
    fetchQA();
  }, [fetchQA]);

  const handleCreate = () => {
    setEditingQA(null);
    setShowForm(true);
  };

  const handleEdit = (qa: CuratedQA) => {
    setEditingQA(qa);
    setShowForm(true);
  };

  const handleSave = async (data: CuratedQACreate | CuratedQAUpdate) => {
    setActionLoading(true);
    try {
      if (editingQA) {
        await updateCuratedQA(editingQA.id, data as CuratedQAUpdate);
        setSuccess('Q&A updated successfully');
      } else {
        await createCuratedQA(data as CuratedQACreate);
        setSuccess('Q&A created successfully');
      }
      await fetchQA();
      setShowForm(false);
    } catch (err) {
      throw err;
    } finally {
      setActionLoading(false);
    }
  };

  const handleToggleEnabled = async (qa: CuratedQA) => {
    setActionLoading(true);
    try {
      await updateCuratedQA(qa.id, { enabled: !qa.enabled });
      setSuccess(`Q&A ${qa.enabled ? 'disabled' : 'enabled'} successfully`);
      await fetchQA();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to toggle status');
    } finally {
      setActionLoading(false);
    }
  };

  const handleDelete = (qa: CuratedQA) => {
    setDeletingQA(qa);
  };

  const handleConfirmDelete = async () => {
    if (!deletingQA) return;
    setActionLoading(true);
    try {
      await deleteCuratedQA(deletingQA.id);
      setSuccess('Q&A deleted successfully');
      await fetchQA();
      setDeletingQA(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete Q&A');
    } finally {
      setActionLoading(false);
    }
  };

  useEffect(() => {
    if (success) {
      const timer = setTimeout(() => setSuccess(null), 3000);
      return () => clearTimeout(timer);
    }
  }, [success]);

  const totalPages = Math.ceil(total / PAGE_SIZE);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-heading font-bold text-gray-900 dark:text-white">
            Curated Q&A
          </h2>
          <p className="text-gray-500 dark:text-gray-400 mt-1">
            Manage pre-approved answers for common questions
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            icon={RefreshCw}
            onClick={() => fetchQA()}
            disabled={loading}
            title="Refresh"
          />
          <Button icon={Plus} onClick={handleCreate}>
            Add Q&A
          </Button>
        </div>
      </div>

      {error && (
        <Alert variant="danger" onClose={() => setError(null)}>
          {error}
        </Alert>
      )}
      {success && (
        <Alert variant="success" onClose={() => setSuccess(null)}>
          {success}
        </Alert>
      )}

      <div className="flex gap-4">
        <div className="max-w-sm flex-1">
          <Input
            placeholder="Search keywords..."
            icon={Search}
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
        <Select
          options={ENABLED_OPTIONS}
          value={enabledFilter}
          onChange={(e) => {
            setEnabledFilter(e.target.value);
            setPage(1);
          }}
        />
      </div>

      {loading ? (
        <div className="py-12 text-center text-gray-500">
          Loading curated Q&A pairs...
        </div>
      ) : qaList.length === 0 ? (
        <div className="py-12 text-center text-gray-500">
          {debouncedSearch || enabledFilter
            ? 'No Q&A pairs match your filters.'
            : 'No curated Q&A pairs yet. Create your first one to get started.'}
        </div>
      ) : (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {qaList.map((qa) => (
              <QACard
                key={qa.id}
                qa={qa}
                onEdit={handleEdit}
                onDelete={handleDelete}
                onToggleEnabled={handleToggleEnabled}
              />
            ))}
          </div>

          {totalPages > 1 && (
            <div className="mt-6">
              <Pagination
                currentPage={page}
                totalPages={totalPages}
                totalItems={total}
                itemsPerPage={PAGE_SIZE}
                onPageChange={setPage}
              />
            </div>
          )}
        </>
      )}

      <QAForm
        isOpen={showForm}
        onClose={() => {
          setShowForm(false);
          setEditingQA(null);
        }}
        onSave={handleSave}
        qa={editingQA}
        isLoading={actionLoading}
      />

      <ConfirmModal
        isOpen={Boolean(deletingQA)}
        onClose={() => setDeletingQA(null)}
        onConfirm={handleConfirmDelete}
        title="Delete Curated Q&A"
        message={`Are you sure you want to delete this Q&A? Keywords: "${deletingQA?.keywords.slice(0, 2).join(', ')}${deletingQA && deletingQA.keywords.length > 2 ? '...' : ''}"`}
        confirmLabel="Delete"
        isLoading={actionLoading}
        variant="danger"
      />
    </div>
  );
}
