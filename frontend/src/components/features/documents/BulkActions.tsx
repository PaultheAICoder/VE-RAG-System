import { Tags, RefreshCw, Trash2 } from 'lucide-react';
import { Button } from '../../ui';

interface BulkActionsProps {
  selectedCount: number;
  onEditTags: () => void;
  onReprocess: () => void;
  onDelete: () => void;
  disabled?: boolean;
}

export function BulkActions({
  selectedCount,
  onEditTags,
  onReprocess,
  onDelete,
  disabled = false,
}: BulkActionsProps) {
  if (selectedCount === 0) {
    return null;
  }

  return (
    <div className="flex items-center gap-4 py-3 px-4 bg-gray-50 dark:bg-gray-800/50 rounded-lg border border-gray-200 dark:border-gray-700">
      <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
        {selectedCount} item{selectedCount !== 1 ? 's' : ''} selected
      </span>

      <div className="flex items-center gap-2 ml-auto">
        <Button
          variant="secondary"
          size="sm"
          icon={Tags}
          onClick={onEditTags}
          disabled={disabled}
        >
          Edit Tags
        </Button>

        <Button
          variant="secondary"
          size="sm"
          icon={RefreshCw}
          onClick={onReprocess}
          disabled={disabled}
        >
          Reprocess
        </Button>

        <Button
          variant="danger"
          size="sm"
          icon={Trash2}
          onClick={onDelete}
          disabled={disabled}
        >
          Delete
        </Button>
      </div>
    </div>
  );
}
