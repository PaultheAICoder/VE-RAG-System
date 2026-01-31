import { useState, useEffect } from 'react';
import { Modal, Button, Input } from '../../ui';
import type { Tag } from '../../../types';
import type { TagCreate, TagUpdate } from '../../../api/tags';

interface TagFormProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: (data: TagCreate | TagUpdate) => Promise<void>;
  tag?: Tag | null; // If provided, we're editing
  isLoading?: boolean;
}

const DEFAULT_COLORS = [
  '#EF4444', // red
  '#F59E0B', // amber
  '#10B981', // emerald
  '#3B82F6', // blue
  '#8B5CF6', // violet
  '#EC4899', // pink
  '#6B7280', // gray
  '#2A9D8F', // primary
];

export function TagForm({ isOpen, onClose, onSave, tag, isLoading = false }: TagFormProps) {
  const [name, setName] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [description, setDescription] = useState('');
  const [color, setColor] = useState('#6B7280');
  const [error, setError] = useState<string | null>(null);

  const isEditing = Boolean(tag);

  // Reset form when modal opens/closes or tag changes
  useEffect(() => {
    if (isOpen) {
      if (tag) {
        setName(tag.name);
        setDisplayName(tag.display_name);
        setDescription(tag.description || '');
        setColor(tag.color || '#6B7280');
      } else {
        setName('');
        setDisplayName('');
        setDescription('');
        setColor('#6B7280');
      }
      setError(null);
    }
  }, [isOpen, tag]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (!name.trim()) {
      setError('Name is required');
      return;
    }
    if (!displayName.trim()) {
      setError('Display name is required');
      return;
    }

    // Validate name format (lowercase, no spaces)
    const nameRegex = /^[a-z0-9_-]+$/;
    if (!isEditing && !nameRegex.test(name)) {
      setError('Name must be lowercase letters, numbers, underscores, or hyphens only');
      return;
    }

    try {
      if (isEditing) {
        await onSave({
          display_name: displayName.trim(),
          description: description.trim() || null,
          color,
        } as TagUpdate);
      } else {
        await onSave({
          name: name.trim().toLowerCase(),
          display_name: displayName.trim(),
          description: description.trim() || null,
          color,
        } as TagCreate);
      }
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save tag');
    }
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} title={isEditing ? 'Edit Tag' : 'Create Tag'} size="md">
      <form onSubmit={handleSubmit} className="space-y-4">
        {error && (
          <div className="p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg text-red-700 dark:text-red-300 text-sm">
            {error}
          </div>
        )}

        <Input
          label="Name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="e.g., hr-policy"
          disabled={isEditing}
          className={isEditing ? 'opacity-50' : ''}
        />
        {isEditing && (
          <p className="text-xs text-gray-500 -mt-2">Tag names cannot be changed after creation</p>
        )}

        <Input
          label="Display Name"
          value={displayName}
          onChange={(e) => setDisplayName(e.target.value)}
          placeholder="e.g., HR Policy"
        />

        <Input
          label="Description (optional)"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="Brief description of this tag"
        />

        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Color
          </label>
          <div className="flex flex-wrap gap-2">
            {DEFAULT_COLORS.map((c) => (
              <button
                key={c}
                type="button"
                onClick={() => setColor(c)}
                className={`w-8 h-8 rounded-full border-2 transition-transform ${
                  color === c
                    ? 'border-gray-800 dark:border-white scale-110'
                    : 'border-transparent hover:scale-105'
                }`}
                style={{ backgroundColor: c }}
              />
            ))}
          </div>
        </div>

        <div className="flex items-center gap-3 pt-4">
          <span
            className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-white text-sm"
            style={{ backgroundColor: color }}
          >
            <span
              className="w-2 h-2 rounded-full bg-white/30"
            />
            {displayName || 'Preview'}
          </span>
        </div>

        <div className="flex justify-end gap-3 pt-4 border-t border-gray-200 dark:border-gray-700">
          <Button variant="secondary" onClick={onClose} disabled={isLoading}>
            Cancel
          </Button>
          <Button type="submit" disabled={isLoading}>
            {isLoading ? 'Saving...' : isEditing ? 'Save Changes' : 'Create Tag'}
          </Button>
        </div>
      </form>
    </Modal>
  );
}
