import { useState } from 'react';
import { AlertTriangle, Trash2 } from 'lucide-react';
import { Modal, Button, Input } from '../../ui';

interface ClearKBModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: (deleteSourceFiles: boolean) => void;
  totalChunks: number;
  totalDocuments: number;
  isLoading?: boolean;
}

export function ClearKBModal({
  isOpen,
  onClose,
  onConfirm,
  totalChunks,
  totalDocuments,
  isLoading = false,
}: ClearKBModalProps) {
  const [confirmText, setConfirmText] = useState('');
  const [deleteSourceFiles, setDeleteSourceFiles] = useState(false);

  const isConfirmValid = confirmText.toUpperCase() === 'CLEAR';

  const handleClose = () => {
    setConfirmText('');
    setDeleteSourceFiles(false);
    onClose();
  };

  const handleConfirm = () => {
    if (isConfirmValid) {
      onConfirm(deleteSourceFiles);
    }
  };

  return (
    <Modal isOpen={isOpen} onClose={handleClose} title="Clear Knowledge Base" size="md">
      <div className="space-y-6">
        {/* Warning */}
        <div className="flex items-start gap-4 p-4 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800">
          <div className="flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center bg-red-100 dark:bg-red-900/30">
            <AlertTriangle size={20} className="text-red-600 dark:text-red-400" />
          </div>
          <div>
            <p className="font-medium text-red-800 dark:text-red-200">
              This action cannot be undone
            </p>
            <p className="text-sm text-red-600 dark:text-red-400 mt-1">
              This will permanently delete all vectors from the knowledge base.
            </p>
          </div>
        </div>

        {/* Stats to be deleted */}
        <div className="p-4 rounded-lg bg-gray-50 dark:bg-gray-800/50">
          <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            This will permanently delete:
          </p>
          <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
            <li className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-red-500" />
              <span>{totalChunks.toLocaleString()} vector chunks</span>
            </li>
            <li className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-red-500" />
              <span>From {totalDocuments} documents</span>
            </li>
          </ul>
        </div>

        {/* Delete source files checkbox */}
        <label className="flex items-start gap-3 cursor-pointer">
          <input
            type="checkbox"
            checked={deleteSourceFiles}
            onChange={(e) => setDeleteSourceFiles(e.target.checked)}
            className="mt-1 w-4 h-4 rounded border-gray-300 text-red-600 focus:ring-red-500"
          />
          <div>
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Also delete source documents from database
            </span>
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
              Removes uploaded files. You will need to re-upload them.
            </p>
          </div>
        </label>

        {/* Confirmation input */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Type <span className="font-mono text-red-600 dark:text-red-400">CLEAR</span> to confirm
          </label>
          <Input
            type="text"
            value={confirmText}
            onChange={(e) => setConfirmText(e.target.value)}
            placeholder="Type CLEAR to proceed..."
            className="font-mono"
          />
        </div>

        {/* Actions */}
        <div className="flex justify-end gap-3 pt-4 border-t border-gray-200 dark:border-gray-700">
          <Button variant="secondary" onClick={handleClose} disabled={isLoading}>
            Cancel
          </Button>
          <Button
            variant="danger"
            icon={Trash2}
            onClick={handleConfirm}
            disabled={!isConfirmValid || isLoading}
          >
            {isLoading ? 'Clearing...' : 'Clear Knowledge Base'}
          </Button>
        </div>
      </div>
    </Modal>
  );
}
