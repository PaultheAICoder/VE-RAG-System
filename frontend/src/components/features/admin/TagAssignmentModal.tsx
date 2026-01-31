import { useState, useEffect } from 'react';
import { Modal, Button, Checkbox } from '../../ui';
import type { Tag, UserWithTags } from '../../../types';

interface TagAssignmentModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: (tagIds: string[]) => Promise<void>;
  user: UserWithTags | null;
  availableTags: Tag[];
  isLoading?: boolean;
}

export function TagAssignmentModal({
  isOpen,
  onClose,
  onSave,
  user,
  availableTags,
  isLoading = false,
}: TagAssignmentModalProps) {
  const [selectedTagIds, setSelectedTagIds] = useState<Set<string>>(new Set());
  const [error, setError] = useState<string | null>(null);

  // Reset selection when modal opens or user changes
  useEffect(() => {
    if (isOpen && user) {
      setSelectedTagIds(new Set(user.tags.map((t) => t.id)));
      setError(null);
    }
  }, [isOpen, user]);

  const handleToggleTag = (tagId: string) => {
    setSelectedTagIds((prev) => {
      const next = new Set(prev);
      if (next.has(tagId)) {
        next.delete(tagId);
      } else {
        next.add(tagId);
      }
      return next;
    });
  };

  const handleSelectAll = () => {
    setSelectedTagIds(new Set(availableTags.map((t) => t.id)));
  };

  const handleSelectNone = () => {
    setSelectedTagIds(new Set());
  };

  const handleSave = async () => {
    setError(null);
    try {
      await onSave(Array.from(selectedTagIds));
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to assign tags');
    }
  };

  if (!user) return null;

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title={`Assign Tags to ${user.display_name}`}
      size="md"
    >
      <div className="space-y-4">
        {error && (
          <div className="p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg text-red-700 dark:text-red-300 text-sm">
            {error}
          </div>
        )}

        <p className="text-sm text-gray-600 dark:text-gray-400">
          Select which tags this user should have access to. Users can only view documents and chat
          about content that matches their assigned tags.
        </p>

        <div className="flex gap-2">
          <button
            type="button"
            onClick={handleSelectAll}
            className="text-sm text-primary hover:underline"
          >
            Select All
          </button>
          <span className="text-gray-300">|</span>
          <button
            type="button"
            onClick={handleSelectNone}
            className="text-sm text-primary hover:underline"
          >
            Select None
          </button>
        </div>

        <div className="max-h-60 overflow-y-auto border border-gray-200 dark:border-gray-700 rounded-lg divide-y divide-gray-200 dark:divide-gray-700">
          {availableTags.length === 0 ? (
            <div className="p-4 text-center text-gray-500">No tags available</div>
          ) : (
            availableTags.map((tag) => (
              <label
                key={tag.id}
                className="flex items-center gap-3 p-3 hover:bg-gray-50 dark:hover:bg-gray-800 cursor-pointer"
              >
                <Checkbox
                  checked={selectedTagIds.has(tag.id)}
                  onChange={() => handleToggleTag(tag.id)}
                />
                <span
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: tag.color || '#6B7280' }}
                />
                <span className="flex-1">
                  <span className="font-medium text-gray-900 dark:text-white">
                    {tag.display_name}
                  </span>
                  <span className="text-sm text-gray-500 dark:text-gray-400 ml-2">
                    ({tag.name})
                  </span>
                </span>
              </label>
            ))
          )}
        </div>

        <div className="text-sm text-gray-500">
          {selectedTagIds.size} of {availableTags.length} tags selected
        </div>

        <div className="flex justify-end gap-3 pt-4 border-t border-gray-200 dark:border-gray-700">
          <Button variant="secondary" onClick={onClose} disabled={isLoading}>
            Cancel
          </Button>
          <Button onClick={handleSave} disabled={isLoading}>
            {isLoading ? 'Saving...' : 'Save Tags'}
          </Button>
        </div>
      </div>
    </Modal>
  );
}
