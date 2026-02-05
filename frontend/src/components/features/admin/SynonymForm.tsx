import { useState, useEffect } from 'react';
import { Modal, Button, Input, TagInput } from '../../ui';
import type { Synonym, SynonymCreate, SynonymUpdate } from '../../../types';

interface SynonymFormProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: (data: SynonymCreate | SynonymUpdate) => Promise<void>;
  synonym?: Synonym | null;
  isLoading?: boolean;
}

export function SynonymForm({
  isOpen,
  onClose,
  onSave,
  synonym,
  isLoading = false,
}: SynonymFormProps) {
  const [term, setTerm] = useState('');
  const [synonyms, setSynonyms] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);

  const isEditing = Boolean(synonym);

  useEffect(() => {
    if (isOpen) {
      if (synonym) {
        setTerm(synonym.term);
        setSynonyms(synonym.synonyms);
      } else {
        setTerm('');
        setSynonyms([]);
      }
      setError(null);
    }
  }, [isOpen, synonym]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    const trimmedTerm = term.trim().toLowerCase();

    if (!trimmedTerm) {
      setError('Term is required');
      return;
    }

    if (synonyms.length === 0) {
      setError('At least one synonym is required');
      return;
    }

    const termRegex = /^[a-z0-9\s\-]+$/;
    if (!termRegex.test(trimmedTerm)) {
      setError('Term must contain only lowercase letters, numbers, spaces, or hyphens');
      return;
    }

    try {
      if (isEditing) {
        await onSave({
          term: trimmedTerm,
          synonyms,
        } as SynonymUpdate);
      } else {
        await onSave({
          term: trimmedTerm,
          synonyms,
        } as SynonymCreate);
      }
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save synonym');
    }
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title={isEditing ? 'Edit Synonym' : 'Create Synonym'}
      size="md"
    >
      <form onSubmit={handleSubmit} className="space-y-4">
        {error && (
          <div className="p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg text-red-700 dark:text-red-300 text-sm">
            {error}
          </div>
        )}

        <Input
          label="Term"
          value={term}
          onChange={(e) => setTerm(e.target.value)}
          placeholder="e.g., pto"
        />
        <p className="text-xs text-gray-500 -mt-2">
          The source term that will be expanded during search
        </p>

        <TagInput
          label="Synonyms"
          value={synonyms}
          onChange={setSynonyms}
          placeholder="Type synonym and press Enter to add"
        />
        <p className="text-xs text-gray-500 -mt-2">
          Related terms that will be included in search queries
        </p>

        <div className="flex justify-end gap-3 pt-4 border-t border-gray-200 dark:border-gray-700">
          <Button variant="secondary" onClick={onClose} disabled={isLoading}>
            Cancel
          </Button>
          <Button type="submit" disabled={isLoading}>
            {isLoading ? 'Saving...' : isEditing ? 'Save Changes' : 'Create Synonym'}
          </Button>
        </div>
      </form>
    </Modal>
  );
}
