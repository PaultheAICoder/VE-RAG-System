import { CheckCircle, XCircle, RefreshCw, SkipForward } from 'lucide-react';
import { Modal, Button, Alert } from '../../ui';
import type { UploadResult } from '../../../types';

interface UploadResultsModalProps {
  isOpen: boolean;
  onClose: () => void;
  results: UploadResult[];
  onViewDocuments: () => void;
}

/**
 * Modal showing upload results after all files are processed.
 */
export function UploadResultsModal({
  isOpen,
  onClose,
  results,
  onViewDocuments,
}: UploadResultsModalProps) {
  const successful = results.filter((r) => r.status === 'success');
  const replaced = results.filter((r) => r.status === 'replaced');
  const failed = results.filter((r) => r.status === 'failed');
  const skipped = results.filter((r) => r.status === 'skipped');

  const totalUploaded = successful.length + replaced.length;
  const hasFailures = failed.length > 0;

  const getStatusIcon = (status: UploadResult['status']) => {
    switch (status) {
      case 'success':
        return <CheckCircle size={16} className="text-green-500" />;
      case 'replaced':
        return <RefreshCw size={16} className="text-amber-500" />;
      case 'failed':
        return <XCircle size={16} className="text-red-500" />;
      case 'skipped':
        return <SkipForward size={16} className="text-gray-400" />;
    }
  };

  const getStatusLabel = (status: UploadResult['status']) => {
    switch (status) {
      case 'success':
        return 'Uploaded';
      case 'replaced':
        return 'Replaced';
      case 'failed':
        return 'Failed';
      case 'skipped':
        return 'Skipped';
    }
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Upload Results" size="lg">
      <div className="space-y-6">
        {/* Summary */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <div className="text-center p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
            <div className="text-2xl font-bold text-green-600 dark:text-green-400">
              {successful.length}
            </div>
            <div className="text-xs text-green-700 dark:text-green-300">Uploaded</div>
          </div>
          <div className="text-center p-3 bg-amber-50 dark:bg-amber-900/20 rounded-lg">
            <div className="text-2xl font-bold text-amber-600 dark:text-amber-400">
              {replaced.length}
            </div>
            <div className="text-xs text-amber-700 dark:text-amber-300">Replaced</div>
          </div>
          <div className="text-center p-3 bg-red-50 dark:bg-red-900/20 rounded-lg">
            <div className="text-2xl font-bold text-red-600 dark:text-red-400">
              {failed.length}
            </div>
            <div className="text-xs text-red-700 dark:text-red-300">Failed</div>
          </div>
          <div className="text-center p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
            <div className="text-2xl font-bold text-gray-600 dark:text-gray-400">
              {skipped.length}
            </div>
            <div className="text-xs text-gray-700 dark:text-gray-300">Skipped</div>
          </div>
        </div>

        {/* Status Message */}
        {hasFailures ? (
          <Alert variant="warning" title="Some uploads failed">
            {failed.length} file{failed.length !== 1 ? 's' : ''} could not be uploaded.
            {totalUploaded > 0 && ` ${totalUploaded} file${totalUploaded !== 1 ? 's' : ''} uploaded successfully.`}
          </Alert>
        ) : totalUploaded > 0 ? (
          <Alert variant="success" title="Upload complete">
            All {totalUploaded} file{totalUploaded !== 1 ? 's' : ''} uploaded successfully.
          </Alert>
        ) : (
          <Alert variant="info" title="No files uploaded">
            All files were skipped.
          </Alert>
        )}

        {/* Failed Files List */}
        {failed.length > 0 && (
          <div>
            <h4 className="text-sm font-medium text-red-600 dark:text-red-400 mb-2">
              Failed Uploads
            </h4>
            <div className="border border-red-200 dark:border-red-800 rounded-lg divide-y divide-red-100 dark:divide-red-800/50">
              {failed.map((result) => (
                <div
                  key={result.filename}
                  className="px-3 py-2 flex items-start gap-2"
                >
                  <XCircle size={16} className="text-red-500 flex-shrink-0 mt-0.5" />
                  <div className="min-w-0 flex-1">
                    <div className="text-sm font-medium text-gray-900 dark:text-gray-100 truncate">
                      {result.filename}
                    </div>
                    {result.error && (
                      <div className="text-xs text-red-600 dark:text-red-400 mt-0.5">
                        {result.error}
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* All Results (collapsible if many) */}
        {results.length > 0 && results.length <= 10 && (
          <div>
            <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              All Files
            </h4>
            <div className="border border-gray-200 dark:border-gray-700 rounded-lg divide-y divide-gray-100 dark:divide-gray-800">
              {results.map((result) => (
                <div
                  key={result.filename}
                  className="px-3 py-2 flex items-center justify-between gap-2"
                >
                  <div className="flex items-center gap-2 min-w-0">
                    {getStatusIcon(result.status)}
                    <span className="text-sm text-gray-900 dark:text-gray-100 truncate">
                      {result.filename}
                    </span>
                  </div>
                  <span
                    className={`
                      text-xs px-2 py-0.5 rounded-full flex-shrink-0
                      ${result.status === 'success' ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300' : ''}
                      ${result.status === 'replaced' ? 'bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300' : ''}
                      ${result.status === 'failed' ? 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300' : ''}
                      ${result.status === 'skipped' ? 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400' : ''}
                    `}
                  >
                    {getStatusLabel(result.status)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Actions */}
        <div className="flex justify-end gap-3 pt-4 border-t border-gray-200 dark:border-gray-700">
          <Button variant="secondary" onClick={onClose}>
            Close
          </Button>
          {totalUploaded > 0 && (
            <Button variant="primary" onClick={onViewDocuments}>
              View Documents
            </Button>
          )}
        </div>
      </div>
    </Modal>
  );
}
