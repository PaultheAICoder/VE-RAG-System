import { useState, useEffect } from 'react';
import { Modal, Button, Alert } from '../../ui';
import type { Tag } from '../../../types';

interface TagEditModalProps {
  isOpen: boolean;
  onClose: () => void;
  tags: Tag[];
  selectedCount: number;
  onSave: (tagIds: string[]) => Promise<void>;
}

export function TagEditModal({
  isOpen,
  onClose,
  tags,
  selectedCount,
  onSave,
}: TagEditModalProps) {
  const [selectedTagIds, setSelectedTagIds] = useState<string[]>([]);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Reset state when modal opens
  useEffect(() => {
    if (isOpen) {
      setSelectedTagIds([]);
      setError(null);
    }
  }, [isOpen]);

  const handleTagToggle = (tagId: string) => {
    setSelectedTagIds((prev) =>
      prev.includes(tagId)
        ? prev.filter((id) => id !== tagId)
        : [...prev, tagId]
    );
    setError(null);
  };

  const handleSave = async () => {
    if (selectedTagIds.length === 0) {
      setError('Please select at least one tag');
      return;
    }

    setIsSaving(true);
    setError(null);

    try {
      await onSave(selectedTagIds);
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update tags');
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Edit Tags" size="md">
      <div className="space-y-6">
        <p className="text-sm text-gray-600 dark:text-gray-400">
          Select tags to assign to {selectedCount} document{selectedCount !== 1 ? 's' : ''}.
          This will replace existing tags.
        </p>

        {/* Tag Selection */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Tags <span className="text-red-500">*</span>
          </label>
          <div className="flex flex-wrap gap-2">
            {tags.map((tag) => (
              <button
                key={tag.id}
                onClick={() => handleTagToggle(tag.id)}
                disabled={isSaving}
                className={`
                  px-3 py-1.5 rounded-full text-sm font-medium transition-all
                  ${selectedTagIds.includes(tag.id)
                    ? 'bg-primary text-white'
                    : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                  }
                  disabled:opacity-50 disabled:cursor-not-allowed
                `}
              >
                {tag.display_name}
              </button>
            ))}
            {tags.length === 0 && (
              <span className="text-sm text-gray-500">No tags available</span>
            )}
          </div>
        </div>

        {/* Error */}
        {error && (
          <Alert variant="danger" onClose={() => setError(null)}>
            {error}
          </Alert>
        )}

        {/* Actions */}
        <div className="flex justify-end gap-3 pt-4 border-t border-gray-200 dark:border-gray-700">
          <Button variant="secondary" onClick={onClose} disabled={isSaving}>
            Cancel
          </Button>
          <Button
            onClick={handleSave}
            disabled={isSaving || selectedTagIds.length === 0}
          >
            {isSaving ? 'Saving...' : 'Save Tags'}
          </Button>
        </div>
      </div>
    </Modal>
  );
}
