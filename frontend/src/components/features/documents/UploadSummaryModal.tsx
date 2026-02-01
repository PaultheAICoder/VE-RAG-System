import { AlertTriangle, Copy, FileText } from 'lucide-react';
import { Modal, Button, Alert } from '../../ui';
import type { DuplicateInfo } from '../../../types';

interface UploadSummaryModalProps {
  isOpen: boolean;
  onClose: () => void;
  duplicates: DuplicateInfo[];
  uniqueFiles: File[];
  onSkipDuplicates: () => void;
  onReplaceAll: () => void;
  onCancel: () => void;
  isLoading?: boolean;
}

/**
 * Modal shown when duplicates are detected during pre-upload check.
 * User can choose to skip duplicates, replace all, or cancel.
 */
export function UploadSummaryModal({
  isOpen,
  onClose,
  duplicates,
  uniqueFiles,
  onSkipDuplicates,
  onReplaceAll,
  onCancel,
  isLoading = false,
}: UploadSummaryModalProps) {
  const hasUnique = uniqueFiles.length > 0;
  const hasDuplicates = duplicates.length > 0;

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Duplicate Files Detected" size="lg">
      <div className="space-y-6">
        {/* Warning Alert */}
        <Alert variant="warning" title="Duplicates Found">
          {duplicates.length} file{duplicates.length !== 1 ? 's' : ''} already exist
          {duplicates.length !== 1 ? '' : 's'} in the system.
          {hasUnique && (
            <span>
              {' '}
              {uniqueFiles.length} unique file{uniqueFiles.length !== 1 ? 's' : ''} will be uploaded.
            </span>
          )}
        </Alert>

        {/* Duplicates List */}
        {hasDuplicates && (
          <div>
            <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2 flex items-center gap-2">
              <Copy size={16} />
              Duplicate Files ({duplicates.length})
            </h4>
            <div className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
              <table className="w-full text-sm">
                <thead className="bg-gray-50 dark:bg-gray-800">
                  <tr>
                    <th className="text-left px-3 py-2 text-gray-600 dark:text-gray-400">
                      Your File
                    </th>
                    <th className="text-left px-3 py-2 text-gray-600 dark:text-gray-400">
                      Existing File
                    </th>
                    <th className="text-left px-3 py-2 text-gray-600 dark:text-gray-400">
                      Uploaded
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                  {duplicates.map((dup) => (
                    <tr key={dup.filename} className="hover:bg-gray-50 dark:hover:bg-gray-800/50">
                      <td className="px-3 py-2 text-gray-900 dark:text-gray-100 truncate max-w-[150px]">
                        {dup.filename}
                      </td>
                      <td className="px-3 py-2 text-gray-600 dark:text-gray-400 truncate max-w-[150px]">
                        {dup.existing_filename}
                      </td>
                      <td className="px-3 py-2 text-gray-500 dark:text-gray-500 text-xs">
                        {new Date(dup.uploaded_at).toLocaleDateString()}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Unique Files List */}
        {hasUnique && (
          <div>
            <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2 flex items-center gap-2">
              <FileText size={16} />
              New Files ({uniqueFiles.length})
            </h4>
            <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-3 max-h-32 overflow-y-auto">
              <ul className="space-y-1">
                {uniqueFiles.map((file) => (
                  <li
                    key={file.name}
                    className="text-sm text-gray-600 dark:text-gray-400 truncate"
                  >
                    {file.name}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        )}

        {/* Actions */}
        <div className="flex flex-col sm:flex-row justify-end gap-3 pt-4 border-t border-gray-200 dark:border-gray-700">
          <Button variant="secondary" onClick={onCancel} disabled={isLoading}>
            Cancel
          </Button>
          {hasUnique && (
            <Button variant="primary" onClick={onSkipDuplicates} disabled={isLoading}>
              {isLoading ? 'Uploading...' : `Upload ${uniqueFiles.length} New File${uniqueFiles.length !== 1 ? 's' : ''}`}
            </Button>
          )}
          <Button
            variant="warning"
            onClick={onReplaceAll}
            disabled={isLoading}
            icon={AlertTriangle}
          >
            {isLoading ? 'Replacing...' : 'Replace All'}
          </Button>
        </div>
      </div>
    </Modal>
  );
}
