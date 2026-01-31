import { useState, useEffect, useCallback } from 'react';
import { AlertTriangle, RotateCcw, RefreshCw } from 'lucide-react';
import { Modal, Button, Alert } from '../../ui';
import { getReindexFailures, retryReindexDocument } from '../../../api/admin';
import type { ReindexFailureInfo } from '../../../types';

interface ReindexFailuresModalProps {
  isOpen: boolean;
  onClose: () => void;
  jobId?: string;
  onRetrySuccess?: () => void;
}

export function ReindexFailuresModal({
  isOpen,
  onClose,
  jobId,
  onRetrySuccess,
}: ReindexFailuresModalProps) {
  const [failures, setFailures] = useState<ReindexFailureInfo[]>([]);
  const [totalFailures, setTotalFailures] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [retryingId, setRetryingId] = useState<string | null>(null);
  const [retrySuccess, setRetrySuccess] = useState<string | null>(null);

  const fetchFailures = useCallback(async () => {
    if (!isOpen) return;
    setLoading(true);
    setError(null);
    try {
      const data = await getReindexFailures(jobId);
      setFailures(data.failures);
      setTotalFailures(data.total_failures);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load failures');
    } finally {
      setLoading(false);
    }
  }, [isOpen, jobId]);

  useEffect(() => {
    if (isOpen) {
      fetchFailures();
    }
  }, [isOpen, fetchFailures]);

  const handleRetry = async (documentId: string) => {
    setRetryingId(documentId);
    setError(null);
    setRetrySuccess(null);
    try {
      await retryReindexDocument(documentId, jobId);
      setRetrySuccess(documentId);
      // Remove from list
      setFailures((prev) => prev.filter((f) => f.document_id !== documentId));
      setTotalFailures((prev) => prev - 1);
      onRetrySuccess?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to retry document');
    } finally {
      setRetryingId(null);
    }
  };

  const truncateId = (id: string) => (id.length > 12 ? `${id.slice(0, 12)}...` : id);

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Failed Documents" size="lg">
      <div className="space-y-4">
        {/* Header */}
        <div className="flex items-center justify-between">
          <p className="text-sm text-gray-500 dark:text-gray-400">
            {totalFailures} document{totalFailures !== 1 ? 's' : ''} failed during reindex
          </p>
          <Button variant="secondary" size="sm" icon={RefreshCw} onClick={fetchFailures} disabled={loading}>
            Refresh
          </Button>
        </div>

        {/* Alerts */}
        {error && (
          <Alert variant="danger" onClose={() => setError(null)}>
            {error}
          </Alert>
        )}
        {retrySuccess && (
          <Alert variant="success" onClose={() => setRetrySuccess(null)}>
            Document marked for retry successfully
          </Alert>
        )}

        {/* Loading state */}
        {loading && (
          <div className="py-8 text-center text-gray-500 dark:text-gray-400">
            Loading failures...
          </div>
        )}

        {/* Empty state */}
        {!loading && failures.length === 0 && (
          <div className="py-8 text-center">
            <div className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-green-100 dark:bg-green-900/30 mb-3">
              <AlertTriangle size={24} className="text-green-600 dark:text-green-400" />
            </div>
            <p className="text-gray-600 dark:text-gray-400">
              No failed documents. All documents processed successfully.
            </p>
          </div>
        )}

        {/* Failures table */}
        {!loading && failures.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left py-2 px-3 font-medium text-gray-700 dark:text-gray-300">
                    Filename
                  </th>
                  <th className="text-left py-2 px-3 font-medium text-gray-700 dark:text-gray-300">
                    Doc ID
                  </th>
                  <th className="text-left py-2 px-3 font-medium text-gray-700 dark:text-gray-300">
                    Error
                  </th>
                  <th className="text-right py-2 px-3 font-medium text-gray-700 dark:text-gray-300">
                    Action
                  </th>
                </tr>
              </thead>
              <tbody>
                {failures.map((failure) => (
                  <tr
                    key={failure.document_id}
                    className="border-b border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-800/50"
                  >
                    <td className="py-2 px-3 text-gray-900 dark:text-white">
                      <span className="truncate block max-w-[200px]" title={failure.filename}>
                        {failure.filename}
                      </span>
                    </td>
                    <td className="py-2 px-3 text-gray-500 dark:text-gray-400 font-mono text-xs">
                      <span title={failure.document_id}>{truncateId(failure.document_id)}</span>
                    </td>
                    <td className="py-2 px-3 text-red-600 dark:text-red-400">
                      <span className="truncate block max-w-[250px]" title={failure.error_message || 'Unknown error'}>
                        {failure.error_message || 'Unknown error'}
                      </span>
                    </td>
                    <td className="py-2 px-3 text-right">
                      <Button
                        variant="secondary"
                        size="sm"
                        icon={RotateCcw}
                        onClick={() => handleRetry(failure.document_id)}
                        disabled={retryingId !== null}
                      >
                        {retryingId === failure.document_id ? 'Retrying...' : 'Retry'}
                      </Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Footer */}
        <div className="flex justify-end pt-4 border-t border-gray-200 dark:border-gray-700">
          <Button variant="secondary" onClick={onClose}>
            Close
          </Button>
        </div>
      </div>
    </Modal>
  );
}
