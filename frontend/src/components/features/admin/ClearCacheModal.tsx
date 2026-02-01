import { ConfirmModal } from './ConfirmModal';

interface ClearCacheModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
  isLoading?: boolean;
  entryCount?: number;
}

export function ClearCacheModal({
  isOpen,
  onClose,
  onConfirm,
  isLoading = false,
  entryCount = 0,
}: ClearCacheModalProps) {
  const message = entryCount > 0
    ? `This will permanently delete all ${entryCount.toLocaleString()} cached query responses. This action cannot be undone.`
    : 'This will permanently delete all cached query responses. This action cannot be undone.';

  return (
    <ConfirmModal
      isOpen={isOpen}
      onClose={onClose}
      onConfirm={onConfirm}
      title="Clear Response Cache"
      message={message}
      confirmLabel="Clear Cache"
      isLoading={isLoading}
      variant="danger"
    />
  );
}
