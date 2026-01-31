import { Copy, Check } from 'lucide-react';
import { useState } from 'react';
import { Modal, Button } from '../../ui';

interface PasswordResetModalProps {
  isOpen: boolean;
  onClose: () => void;
  userName: string;
  temporaryPassword: string;
}

export function PasswordResetModal({
  isOpen,
  onClose,
  userName,
  temporaryPassword,
}: PasswordResetModalProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(temporaryPassword);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Password Reset" size="sm">
      <div className="space-y-4">
        <p className="text-gray-600 dark:text-gray-400">
          A temporary password has been generated for <strong>{userName}</strong>. They will be
          required to change it on their next login.
        </p>

        <div className="p-4 bg-gray-100 dark:bg-gray-800 rounded-lg">
          <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
            Temporary Password
          </label>
          <div className="flex items-center gap-2">
            <code className="flex-1 text-lg font-mono text-gray-900 dark:text-white">
              {temporaryPassword}
            </code>
            <button
              onClick={handleCopy}
              className="p-2 text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700 rounded transition-colors"
              title="Copy to clipboard"
            >
              {copied ? <Check size={18} className="text-green-500" /> : <Copy size={18} />}
            </button>
          </div>
        </div>

        <p className="text-sm text-amber-600 dark:text-amber-400">
          Please share this password securely with the user. It will not be shown again.
        </p>

        <div className="flex justify-end pt-4 border-t border-gray-200 dark:border-gray-700">
          <Button onClick={onClose}>Done</Button>
        </div>
      </div>
    </Modal>
  );
}
