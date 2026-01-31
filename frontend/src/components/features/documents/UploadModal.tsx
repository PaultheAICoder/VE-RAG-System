import { useState, useCallback } from 'react';
import { Modal, Button, Alert } from '../../ui';
import { UploadDropZone } from './UploadDropZone';
import { UploadQueue, type QueuedFile, type UploadStatus } from './UploadQueue';
import { uploadDocument } from '../../../api/documents';
import type { Tag } from '../../../types';

interface UploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  tags: Tag[];
  onUploadComplete: () => void;
}

function generateId(): string {
  return Math.random().toString(36).substring(2, 9);
}

export function UploadModal({
  isOpen,
  onClose,
  tags,
  onUploadComplete,
}: UploadModalProps) {
  const [queuedFiles, setQueuedFiles] = useState<QueuedFile[]>([]);
  const [selectedTagIds, setSelectedTagIds] = useState<string[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFilesSelected = useCallback((files: File[]) => {
    const newFiles: QueuedFile[] = files.map((file) => ({
      id: generateId(),
      file,
      status: 'queued' as UploadStatus,
      progress: 0,
    }));
    setQueuedFiles((prev) => [...prev, ...newFiles]);
    setError(null);
  }, []);

  const handleRemoveFile = useCallback((id: string) => {
    setQueuedFiles((prev) => prev.filter((f) => f.id !== id));
  }, []);

  const handleTagToggle = useCallback((tagId: string) => {
    setSelectedTagIds((prev) =>
      prev.includes(tagId)
        ? prev.filter((id) => id !== tagId)
        : [...prev, tagId]
    );
    setError(null);
  }, []);

  const handleUploadAll = async () => {
    if (selectedTagIds.length === 0) {
      setError('Please select at least one tag');
      return;
    }

    const filesToUpload = queuedFiles.filter((f) => f.status === 'queued');
    if (filesToUpload.length === 0) {
      return;
    }

    setIsUploading(true);
    setError(null);

    for (const qf of filesToUpload) {
      // Mark as uploading
      setQueuedFiles((prev) =>
        prev.map((f) =>
          f.id === qf.id ? { ...f, status: 'uploading' as UploadStatus, progress: 0 } : f
        )
      );

      try {
        await uploadDocument(qf.file, selectedTagIds, (progress) => {
          setQueuedFiles((prev) =>
            prev.map((f) => (f.id === qf.id ? { ...f, progress } : f))
          );
        });

        // Mark as done
        setQueuedFiles((prev) =>
          prev.map((f) =>
            f.id === qf.id ? { ...f, status: 'done' as UploadStatus, progress: 100 } : f
          )
        );
      } catch (err) {
        // Mark as failed
        setQueuedFiles((prev) =>
          prev.map((f) =>
            f.id === qf.id
              ? {
                  ...f,
                  status: 'failed' as UploadStatus,
                  error: err instanceof Error ? err.message : 'Upload failed',
                }
              : f
          )
        );
      }
    }

    setIsUploading(false);
    onUploadComplete();

    // Close modal immediately - document table shows processing status
    setQueuedFiles([]);
    setSelectedTagIds([]);
    setError(null);
    onClose();
  };

  const handleClose = () => {
    if (!isUploading) {
      setQueuedFiles([]);
      setSelectedTagIds([]);
      setError(null);
      onClose();
    }
  };

  const hasQueuedFiles = queuedFiles.some((f) => f.status === 'queued');
  const allDone = queuedFiles.length > 0 && queuedFiles.every((f) => f.status === 'done');

  return (
    <Modal isOpen={isOpen} onClose={handleClose} title="Upload Documents" size="lg">
      <div className="space-y-6">
        {/* Drop Zone */}
        <UploadDropZone
          onFilesSelected={handleFilesSelected}
          disabled={isUploading}
        />

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
                disabled={isUploading}
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

        {/* Upload Queue */}
        <UploadQueue files={queuedFiles} onRemove={handleRemoveFile} />

        {/* Actions */}
        <div className="flex justify-end gap-3 pt-4 border-t border-gray-200 dark:border-gray-700">
          <Button variant="secondary" onClick={handleClose} disabled={isUploading}>
            {allDone ? 'Close' : 'Cancel'}
          </Button>
          {!allDone && (
            <Button
              onClick={handleUploadAll}
              disabled={!hasQueuedFiles || isUploading || selectedTagIds.length === 0}
            >
              {isUploading ? 'Uploading...' : 'Upload All'}
            </Button>
          )}
        </div>
      </div>
    </Modal>
  );
}
