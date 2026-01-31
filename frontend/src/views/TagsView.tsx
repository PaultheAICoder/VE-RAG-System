import { useState, useEffect, useCallback } from 'react';
import { Plus, Pencil, Trash2 } from 'lucide-react';
import { Button, Alert, Card } from '../components/ui';
import { TagForm, ConfirmModal } from '../components/features/admin';
import { listTags, createTag, updateTag, deleteTag } from '../api/tags';
import type { TagCreate, TagUpdate } from '../api/tags';
import type { Tag } from '../types';

export function TagsView() {
  // Data state
  const [tags, setTags] = useState<Tag[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Modal state
  const [showTagForm, setShowTagForm] = useState(false);
  const [editingTag, setEditingTag] = useState<Tag | null>(null);
  const [deletingTag, setDeletingTag] = useState<Tag | null>(null);
  const [actionLoading, setActionLoading] = useState(false);

  // Fetch tags
  const fetchTags = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await listTags();
      setTags(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load tags');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchTags();
  }, [fetchTags]);

  // Handle create tag
  const handleCreate = () => {
    setEditingTag(null);
    setShowTagForm(true);
  };

  // Handle edit tag
  const handleEdit = (tag: Tag) => {
    setEditingTag(tag);
    setShowTagForm(true);
  };

  // Handle save tag (create or update)
  const handleSaveTag = async (data: TagCreate | TagUpdate) => {
    setActionLoading(true);
    try {
      if (editingTag) {
        await updateTag(editingTag.id, data as TagUpdate);
      } else {
        await createTag(data as TagCreate);
      }
      await fetchTags();
      setShowTagForm(false);
    } catch (err) {
      throw err;
    } finally {
      setActionLoading(false);
    }
  };

  // Handle delete tag
  const handleDelete = (tag: Tag) => {
    setDeletingTag(tag);
  };

  const handleConfirmDelete = async () => {
    if (!deletingTag) return;
    setActionLoading(true);
    try {
      await deleteTag(deletingTag.id);
      await fetchTags();
      setDeletingTag(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete tag');
    } finally {
      setActionLoading(false);
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-heading font-bold text-gray-900 dark:text-white">Tags</h1>
          <p className="text-gray-500 dark:text-gray-400 mt-1">
            Manage tags for document access control
          </p>
        </div>
        <Button icon={Plus} onClick={handleCreate}>
          Add Tag
        </Button>
      </div>

      {/* Error Alert */}
      {error && (
        <Alert variant="danger" onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Tags Table */}
      <Card>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-200 dark:border-gray-700">
                <th className="text-left py-3 px-4 text-sm font-semibold text-gray-600 dark:text-gray-300">
                  Tag
                </th>
                <th className="text-left py-3 px-4 text-sm font-semibold text-gray-600 dark:text-gray-300">
                  Display Name
                </th>
                <th className="text-left py-3 px-4 text-sm font-semibold text-gray-600 dark:text-gray-300">
                  Description
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
                    Loading tags...
                  </td>
                </tr>
              ) : tags.length === 0 ? (
                <tr>
                  <td colSpan={4} className="py-8 text-center text-gray-500">
                    No tags found. Create your first tag to get started.
                  </td>
                </tr>
              ) : (
                tags.map((tag) => (
                  <tr
                    key={tag.id}
                    className="border-b border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-800/50"
                  >
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-2">
                        <span
                          className="w-3 h-3 rounded-full"
                          style={{ backgroundColor: tag.color || '#6B7280' }}
                        />
                        <code className="text-sm text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-800 px-2 py-0.5 rounded">
                          {tag.name}
                        </code>
                        {tag.is_system && (
                          <span className="text-xs bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 px-2 py-0.5 rounded">
                            System
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="py-3 px-4 text-gray-900 dark:text-white">{tag.display_name}</td>
                    <td className="py-3 px-4 text-gray-500 dark:text-gray-400 max-w-xs truncate">
                      {tag.description || '-'}
                    </td>
                    <td className="py-3 px-4">
                      <div className="flex items-center justify-end gap-2">
                        <button
                          onClick={() => handleEdit(tag)}
                          className="p-1.5 text-gray-400 hover:text-primary hover:bg-primary/10 rounded transition-colors"
                          title="Edit tag"
                          disabled={tag.is_system}
                        >
                          <Pencil size={16} />
                        </button>
                        {!tag.is_system && (
                          <button
                            onClick={() => handleDelete(tag)}
                            className="p-1.5 text-gray-400 hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20 rounded transition-colors"
                            title="Delete tag"
                          >
                            <Trash2 size={16} />
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Tag Form Modal */}
      <TagForm
        isOpen={showTagForm}
        onClose={() => {
          setShowTagForm(false);
          setEditingTag(null);
        }}
        onSave={handleSaveTag}
        tag={editingTag}
        isLoading={actionLoading}
      />

      {/* Delete Confirmation Modal */}
      <ConfirmModal
        isOpen={Boolean(deletingTag)}
        onClose={() => setDeletingTag(null)}
        onConfirm={handleConfirmDelete}
        title="Delete Tag"
        message={`Are you sure you want to delete the tag "${deletingTag?.display_name}"? This will remove access control for any documents and users using this tag.`}
        confirmLabel="Delete"
        isLoading={actionLoading}
        variant="danger"
      />
    </div>
  );
}
