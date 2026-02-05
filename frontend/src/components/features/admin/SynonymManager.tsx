import { useState, useEffect, useCallback } from 'react';
import { Plus, Pencil, Trash2, Search, ToggleLeft, ToggleRight } from 'lucide-react';
import { Button, Alert, Card, Input, Pagination } from '../../ui';
import { SynonymForm } from './SynonymForm';
import { ConfirmModal } from './ConfirmModal';
import {
  listSynonyms,
  createSynonym,
  updateSynonym,
  deleteSynonym,
  toggleSynonymStatus,
} from '../../../api/synonyms';
import type { Synonym, SynonymCreate, SynonymUpdate } from '../../../types';

const PAGE_SIZE = 10;

export function SynonymManager() {
  const [synonyms, setSynonyms] = useState<Synonym[]>([]);
  const [total, setTotal] = useState(0);
  const [currentPage, setCurrentPage] = useState(1);
  const [searchTerm, setSearchTerm] = useState('');
  const [debouncedSearch, setDebouncedSearch] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [showForm, setShowForm] = useState(false);
  const [editingSynonym, setEditingSynonym] = useState<Synonym | null>(null);
  const [deletingSynonym, setDeletingSynonym] = useState<Synonym | null>(null);
  const [actionLoading, setActionLoading] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedSearch(searchTerm);
      setCurrentPage(1);
    }, 300);
    return () => clearTimeout(timer);
  }, [searchTerm]);

  const fetchSynonyms = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await listSynonyms({
        page: currentPage,
        page_size: PAGE_SIZE,
        search: debouncedSearch || undefined,
      });
      setSynonyms(response.synonyms);
      setTotal(response.total);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load synonyms');
    } finally {
      setLoading(false);
    }
  }, [currentPage, debouncedSearch]);

  useEffect(() => {
    fetchSynonyms();
  }, [fetchSynonyms]);

  const handleCreate = () => {
    setEditingSynonym(null);
    setShowForm(true);
  };

  const handleEdit = (synonym: Synonym) => {
    setEditingSynonym(synonym);
    setShowForm(true);
  };

  const handleSave = async (data: SynonymCreate | SynonymUpdate) => {
    setActionLoading(true);
    try {
      if (editingSynonym) {
        await updateSynonym(editingSynonym.id, data as SynonymUpdate);
      } else {
        await createSynonym(data as SynonymCreate);
      }
      await fetchSynonyms();
      setShowForm(false);
    } catch (err) {
      throw err;
    } finally {
      setActionLoading(false);
    }
  };

  const handleDelete = (synonym: Synonym) => {
    setDeletingSynonym(synonym);
  };

  const handleConfirmDelete = async () => {
    if (!deletingSynonym) return;
    setActionLoading(true);
    try {
      await deleteSynonym(deletingSynonym.id);
      await fetchSynonyms();
      setDeletingSynonym(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete synonym');
    } finally {
      setActionLoading(false);
    }
  };

  const handleToggleStatus = async (synonym: Synonym) => {
    try {
      await toggleSynonymStatus(synonym.id, !synonym.enabled);
      await fetchSynonyms();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to toggle status');
    }
  };

  const totalPages = Math.ceil(total / PAGE_SIZE);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-heading font-semibold text-gray-900 dark:text-white">
            Query Synonyms
          </h2>
          <p className="text-gray-500 dark:text-gray-400 text-sm mt-1">
            Define synonym mappings for query expansion in RAG searches
          </p>
        </div>
        <Button icon={Plus} onClick={handleCreate}>
          Add Synonym
        </Button>
      </div>

      <div className="max-w-md">
        <Input
          icon={Search}
          placeholder="Search terms or synonyms..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
        />
      </div>

      {error && (
        <Alert variant="danger" onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <Card>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-200 dark:border-gray-700">
                <th className="text-left py-3 px-4 text-sm font-semibold text-gray-600 dark:text-gray-300">
                  Term
                </th>
                <th className="text-left py-3 px-4 text-sm font-semibold text-gray-600 dark:text-gray-300">
                  Synonyms
                </th>
                <th className="text-center py-3 px-4 text-sm font-semibold text-gray-600 dark:text-gray-300">
                  Status
                </th>
                <th className="text-right py-3 px-4 text-sm font-semibold text-gray-600 dark:text-gray-300">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan={4} className="py-8 text-center text-gray-500">
                    Loading synonyms...
                  </td>
                </tr>
              ) : synonyms.length === 0 ? (
                <tr>
                  <td colSpan={4} className="py-8 text-center text-gray-500">
                    {debouncedSearch
                      ? 'No synonyms match your search.'
                      : 'No synonyms found. Create your first synonym mapping.'}
                  </td>
                </tr>
              ) : (
                synonyms.map((synonym) => (
                  <tr
                    key={synonym.id}
                    className="border-b border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-800/50"
                  >
                    <td className="py-3 px-4">
                      <code className="text-sm text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-800 px-2 py-0.5 rounded">
                        {synonym.term}
                      </code>
                    </td>
                    <td className="py-3 px-4">
                      <div className="flex flex-wrap gap-1">
                        {synonym.synonyms.map((s, idx) => (
                          <span
                            key={idx}
                            className="inline-block px-2 py-0.5 text-xs rounded-full bg-primary/10 text-primary dark:bg-primary/20"
                          >
                            {s}
                          </span>
                        ))}
                      </div>
                    </td>
                    <td className="py-3 px-4 text-center">
                      <button
                        onClick={() => handleToggleStatus(synonym)}
                        className={`
                          inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium transition-colors
                          ${synonym.enabled
                            ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400 hover:bg-green-200 dark:hover:bg-green-900/50'
                            : 'bg-gray-100 text-gray-500 dark:bg-gray-800 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700'
                          }
                        `}
                      >
                        {synonym.enabled ? (
                          <>
                            <ToggleRight size={14} />
                            Enabled
                          </>
                        ) : (
                          <>
                            <ToggleLeft size={14} />
                            Disabled
                          </>
                        )}
                      </button>
                    </td>
                    <td className="py-3 px-4">
                      <div className="flex items-center justify-end gap-2">
                        <button
                          onClick={() => handleEdit(synonym)}
                          className="p-1.5 text-gray-400 hover:text-primary hover:bg-primary/10 rounded transition-colors"
                          title="Edit synonym"
                        >
                          <Pencil size={16} />
                        </button>
                        <button
                          onClick={() => handleDelete(synonym)}
                          className="p-1.5 text-gray-400 hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20 rounded transition-colors"
                          title="Delete synonym"
                        >
                          <Trash2 size={16} />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        {totalPages > 1 && (
          <div className="px-4 py-3 border-t border-gray-200 dark:border-gray-700">
            <Pagination
              currentPage={currentPage}
              totalPages={totalPages}
              totalItems={total}
              itemsPerPage={PAGE_SIZE}
              onPageChange={setCurrentPage}
            />
          </div>
        )}
      </Card>

      <SynonymForm
        isOpen={showForm}
        onClose={() => {
          setShowForm(false);
          setEditingSynonym(null);
        }}
        onSave={handleSave}
        synonym={editingSynonym}
        isLoading={actionLoading}
      />

      <ConfirmModal
        isOpen={Boolean(deletingSynonym)}
        onClose={() => setDeletingSynonym(null)}
        onConfirm={handleConfirmDelete}
        title="Delete Synonym"
        message={`Are you sure you want to delete the synonym mapping for "${deletingSynonym?.term}"? This will affect query expansion in RAG searches.`}
        confirmLabel="Delete"
        isLoading={actionLoading}
        variant="danger"
      />
    </div>
  );
}
