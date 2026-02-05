import { useState, useEffect } from 'react';
import { Modal, Button, Input, TagInput, RichTextEditor } from '../../ui';
import { sanitizeHtml } from '../../../utils/sanitize';
import type { CuratedQA, CuratedQACreate, CuratedQAUpdate } from '../../../types';

interface QAFormProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: (data: CuratedQACreate | CuratedQAUpdate) => Promise<void>;
  qa?: CuratedQA | null;
  isLoading?: boolean;
}

export function QAForm({
  isOpen,
  onClose,
  onSave,
  qa,
  isLoading = false,
}: QAFormProps) {
  const [keywords, setKeywords] = useState<string[]>([]);
  const [answer, setAnswer] = useState('');
  const [sourceReference, setSourceReference] = useState('');
  const [confidence, setConfidence] = useState(85);
  const [priority, setPriority] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const isEditing = Boolean(qa);

  useEffect(() => {
    if (isOpen) {
      if (qa) {
        setKeywords(qa.keywords);
        setAnswer(qa.answer);
        setSourceReference(qa.source_reference);
        setConfidence(qa.confidence);
        setPriority(qa.priority);
      } else {
        setKeywords([]);
        setAnswer('');
        setSourceReference('');
        setConfidence(85);
        setPriority(0);
      }
      setError(null);
    }
  }, [isOpen, qa]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (keywords.length === 0) {
      setError('At least one keyword is required');
      return;
    }

    if (!answer.trim() || answer === '<p></p>') {
      setError('Answer is required');
      return;
    }

    if (!sourceReference.trim()) {
      setError('Source reference is required for compliance');
      return;
    }

    if (confidence < 0 || confidence > 100) {
      setError('Confidence must be between 0 and 100');
      return;
    }

    if (priority < 0) {
      setError('Priority must be 0 or greater');
      return;
    }

    try {
      const sanitizedAnswer = sanitizeHtml(answer);

      if (isEditing) {
        await onSave({
          keywords,
          answer: sanitizedAnswer,
          source_reference: sourceReference.trim(),
          confidence,
          priority,
        } as CuratedQAUpdate);
      } else {
        await onSave({
          keywords,
          answer: sanitizedAnswer,
          source_reference: sourceReference.trim(),
          confidence,
          priority,
        } as CuratedQACreate);
      }
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save Q&A');
    }
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title={isEditing ? 'Edit Curated Q&A' : 'Create Curated Q&A'}
      size="lg"
    >
      <form onSubmit={handleSubmit} className="space-y-4">
        {error && (
          <div className="p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg text-red-700 dark:text-red-300 text-sm">
            {error}
          </div>
        )}

        <TagInput
          label="Keywords"
          value={keywords}
          onChange={setKeywords}
          placeholder="Type keyword and press Enter to add"
        />
        <p className="text-xs text-gray-500 -mt-2">
          Enter keywords/phrases that should trigger this curated response
        </p>

        <Input
          label="Source Reference"
          value={sourceReference}
          onChange={(e) => setSourceReference(e.target.value)}
          placeholder="e.g., HR Policy Manual Section 3.2"
        />
        <p className="text-xs text-gray-500 -mt-2">
          Required for compliance - cite the authoritative source
        </p>

        <RichTextEditor
          label="Answer"
          value={answer}
          onChange={setAnswer}
          placeholder="Enter the curated answer with formatting..."
        />

        <div className="grid grid-cols-2 gap-4">
          <div>
            <Input
              label="Confidence (%)"
              type="number"
              value={confidence}
              onChange={(e) => setConfidence(Number(e.target.value))}
            />
            <p className="text-xs text-gray-500 mt-1">0-100, default 85</p>
          </div>
          <div>
            <Input
              label="Priority"
              type="number"
              value={priority}
              onChange={(e) => setPriority(Number(e.target.value))}
            />
            <p className="text-xs text-gray-500 mt-1">Higher = matched first</p>
          </div>
        </div>

        <div className="flex justify-end gap-3 pt-4 border-t border-gray-200 dark:border-gray-700">
          <Button variant="secondary" onClick={onClose} disabled={isLoading}>
            Cancel
          </Button>
          <Button type="submit" disabled={isLoading}>
            {isLoading ? 'Saving...' : isEditing ? 'Save Changes' : 'Create Q&A'}
          </Button>
        </div>
      </form>
    </Modal>
  );
}
