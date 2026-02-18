import { useState, useCallback } from 'react';
import { Modal, Button, Alert } from '../../ui';
import { UploadDropZone } from './UploadDropZone';
import { UploadQueue, type QueuedFile, type UploadStatus } from './UploadQueue';
import { UploadSummaryModal } from './UploadSummaryModal';
import { UploadResultsModal } from './UploadResultsModal';
import { uploadDocument, checkDuplicates } from '../../../api/documents';
import type { Tag, CheckDuplicatesResponse, UploadResult, UploadErrorResponse } from '../../../types';

interface UploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  tags: Tag[];
  onUploadComplete: () => void;
  autoTaggingEnabled?: boolean;
}

function generateId(): string {
  return Math.random().toString(36).substring(2, 9);
}

export function UploadModal({
  isOpen,
  onClose,
  tags,
  onUploadComplete,
  autoTaggingEnabled = false,
}: UploadModalProps) {
  const [queuedFiles, setQueuedFiles] = useState<QueuedFile[]>([]);
  const [selectedTagIds, setSelectedTagIds] = useState<string[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Duplicate check state
  const [showSummary, setShowSummary] = useState(false);
  const [duplicateCheck, setDuplicateCheck] = useState<CheckDuplicatesResponse | null>(null);

  // Results state
  const [showResults, setShowResults] = useState(false);
  const [uploadResults, setUploadResults] = useState<UploadResult[]>([]);
  const [selectionSkippedResults, setSelectionSkippedResults] = useState<UploadResult[]>([]);

  const handleFilesSelected = useCallback((files: File[], skippedFiles: File[] = []) => {
    if (files.length > 0) {
      const newFiles: QueuedFile[] = files.map((file) => ({
        id: generateId(),
        file,
        status: 'queued' as UploadStatus,
        progress: 0,
      }));
      setQueuedFiles((prev) => [...prev, ...newFiles]);
    }

    if (skippedFiles.length > 0) {
      const skippedResults: UploadResult[] = skippedFiles.map((file) => ({
        filename: file.name,
        status: 'skipped' as const,
        error: 'Unsupported file type',
      }));
      setSelectionSkippedResults((prev) => [...prev, ...skippedResults]);
    }

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

  // Upload files with optional replace mode
  const uploadFiles = async (files: QueuedFile[], replaceMode: boolean) => {
    const results: UploadResult[] = [];

    for (const qf of files) {
      // Mark as uploading
      setQueuedFiles((prev) =>
        prev.map((f) =>
          f.id === qf.id ? { ...f, status: 'uploading' as UploadStatus, progress: 0 } : f
        )
      );

      try {
        const doc = await uploadDocument(qf.file, selectedTagIds, (progress) => {
          setQueuedFiles((prev) =>
            prev.map((f) => (f.id === qf.id ? { ...f, progress } : f))
          );
        }, replaceMode);

        // Mark as done
        setQueuedFiles((prev) =>
          prev.map((f) =>
            f.id === qf.id ? { ...f, status: 'done' as UploadStatus, progress: 100 } : f
          )
        );

        results.push({
          filename: qf.file.name,
          status: replaceMode ? 'replaced' : 'success',
          documentId: doc.id,
        });
      } catch (err) {
        // Extract structured error if available
        const errorResponse = (err as Error & { response?: UploadErrorResponse }).response;
        const errorMessage = err instanceof Error ? err.message : 'Upload failed';

        // Mark as failed
        setQueuedFiles((prev) =>
          prev.map((f) =>
            f.id === qf.id
              ? { ...f, status: 'failed' as UploadStatus, error: errorMessage }
              : f
          )
        );

        results.push({
          filename: qf.file.name,
          status: 'failed',
          error: errorResponse?.error_code === 'DUPLICATE_FILE'
            ? `Duplicate of "${errorResponse.existing_filename}"`
            : errorMessage,
        });
      }
    }

    return results;
  };

  const handleUploadAll = async () => {
    if (selectedTagIds.length === 0 && !autoTaggingEnabled) {
      setError('Please select at least one tag');
      return;
    }

    const filesToUpload = queuedFiles.filter((f) => f.status === 'queued');
    if (filesToUpload.length === 0) {
      return;
    }

    setIsUploading(true);
    setError(null);

    try {
      // Pre-upload duplicate check
      const filenames = filesToUpload.map((f) => f.file.name);
      const checkResult = await checkDuplicates(filenames);

      if (checkResult.duplicates.length > 0) {
        setDuplicateCheck(checkResult);
        setShowSummary(true);
        setIsUploading(false);
        return;
      }

      // No duplicates - proceed with upload
      const results = await uploadFiles(filesToUpload, false);
      setUploadResults([...results, ...selectionSkippedResults]);
      setIsUploading(false);
      onUploadComplete();
      setShowResults(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to check duplicates');
      setIsUploading(false);
    }
  };

  const handleSkipDuplicates = async () => {
    if (!duplicateCheck) return;

    setShowSummary(false);
    setIsUploading(true);

    // Get unique files only
    const uniqueFilenames = new Set(duplicateCheck.unique);
    const filesToUpload = queuedFiles.filter(
      (f) => f.status === 'queued' && uniqueFilenames.has(f.file.name)
    );

    // Mark skipped files
    const skippedResults: UploadResult[] = duplicateCheck.duplicates.map((dup) => ({
      filename: dup.filename,
      status: 'skipped' as const,
    }));

    const uploadedResults = await uploadFiles(filesToUpload, false);
    setUploadResults([...uploadedResults, ...skippedResults, ...selectionSkippedResults]);
    setIsUploading(false);
    onUploadComplete();
    setShowResults(true);
  };

  const handleReplaceAll = async () => {
    if (!duplicateCheck) return;

    setShowSummary(false);
    setIsUploading(true);

    // Upload all files with replace=true
    const filesToUpload = queuedFiles.filter((f) => f.status === 'queued');
    const results = await uploadFiles(filesToUpload, true);
    setUploadResults([...results, ...selectionSkippedResults]);
    setIsUploading(false);
    onUploadComplete();
    setShowResults(true);
  };

  const handleCancelSummary = () => {
    setShowSummary(false);
    setDuplicateCheck(null);
  };

  const handleViewDocuments = () => {
    setShowResults(false);
    resetAndClose();
  };

  const handleResultsClose = () => {
    setShowResults(false);
    resetAndClose();
  };

  const resetAndClose = () => {
    setQueuedFiles([]);
    setSelectedTagIds([]);
    setError(null);
    setDuplicateCheck(null);
    setUploadResults([]);
    setSelectionSkippedResults([]);
    onClose();
  };

  const handleClose = () => {
    if (!isUploading) {
      resetAndClose();
    }
  };

  const hasQueuedFiles = queuedFiles.some((f) => f.status === 'queued');

  // Get unique files for summary modal
  const getUniqueFiles = (): File[] => {
    if (!duplicateCheck) return [];
    const uniqueFilenames = new Set(duplicateCheck.unique);
    return queuedFiles
      .filter((qf) => uniqueFilenames.has(qf.file.name))
      .map((qf) => qf.file);
  };

  return (
    <>
      <Modal isOpen={isOpen && !showSummary && !showResults} onClose={handleClose} title="Upload Documents" size="lg">
        <div className="space-y-6">
          {/* Drop Zone */}
          <UploadDropZone
            onFilesSelected={handleFilesSelected}
            disabled={isUploading}
          />

          {/* Tag Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Tags {!autoTaggingEnabled && <span className="text-red-500">*</span>}
              {autoTaggingEnabled && (
                <span className="text-xs text-gray-500 dark:text-gray-400 ml-2">(optional — auto-tagging enabled)</span>
              )}
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

          {selectionSkippedResults.length > 0 && (
            <Alert variant="warning">
              {selectionSkippedResults.length} unsupported file
              {selectionSkippedResults.length !== 1 ? 's were' : ' was'} skipped.
            </Alert>
          )}

          {/* Upload Queue */}
          <UploadQueue files={queuedFiles} onRemove={handleRemoveFile} />

          {/* Actions */}
          <div className="flex justify-end gap-3 pt-4 border-t border-gray-200 dark:border-gray-700">
            <Button variant="secondary" onClick={handleClose} disabled={isUploading}>
              Cancel
            </Button>
            <Button
              onClick={handleUploadAll}
              disabled={!hasQueuedFiles || isUploading || (!autoTaggingEnabled && selectedTagIds.length === 0)}
            >
              {isUploading ? 'Checking...' : 'Upload All'}
            </Button>
          </div>
        </div>
      </Modal>

      {/* Summary Modal */}
      <UploadSummaryModal
        isOpen={showSummary}
        onClose={handleCancelSummary}
        duplicates={duplicateCheck?.duplicates ?? []}
        uniqueFiles={getUniqueFiles()}
        onSkipDuplicates={handleSkipDuplicates}
        onReplaceAll={handleReplaceAll}
        onCancel={handleCancelSummary}
        isLoading={isUploading}
      />

      {/* Results Modal */}
      <UploadResultsModal
        isOpen={showResults}
        onClose={handleResultsClose}
        results={uploadResults}
        onViewDocuments={handleViewDocuments}
      />
    </>
  );
}
