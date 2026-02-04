import { MessageSquare, Trash2 } from 'lucide-react';
import type { ChatSession } from '../../../types';
import { Checkbox } from '../../ui';

interface SessionItemProps {
  session: ChatSession;
  isActive: boolean;
  onClick: () => void;
  /** Show delete button (admin only) */
  showDelete?: boolean;
  /** Callback when delete button is clicked */
  onDelete?: (sessionId: string) => void;
  /** Show selection checkbox (for bulk operations) */
  showSelect?: boolean;
  /** Whether this item is selected */
  isSelected?: boolean;
  /** Callback when selection changes */
  onSelect?: (sessionId: string, selected: boolean) => void;
}

function formatRelativeDate(dateString: string): string {
  const date = new Date(dateString);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffDays = Math.floor(diffMs / 86400000);

  if (diffDays === 0) return 'Today';
  if (diffDays === 1) return 'Yesterday';
  if (diffDays < 7) return `${diffDays} days ago`;

  return date.toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
  });
}

export function SessionItem({
  session,
  isActive,
  onClick,
  showDelete = false,
  onDelete,
  showSelect = false,
  isSelected = false,
  onSelect,
}: SessionItemProps) {
  const title = session.title || 'New Chat';
  const preview = session.last_message_preview || 'No messages yet';

  const handleDelete = (e: React.MouseEvent) => {
    e.stopPropagation();
    onDelete?.(session.id);
  };

  const handleSelectChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.stopPropagation();
    onSelect?.(session.id, e.target.checked);
  };

  const handleCheckboxClick = (e: React.MouseEvent) => {
    e.stopPropagation();
  };

  return (
    <div
      onClick={onClick}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onClick();
        }
      }}
      className={`
        w-full text-left px-3 py-3 rounded-lg transition-colors cursor-pointer
        ${isActive
          ? 'bg-primary/10 border-l-4 border-primary'
          : 'hover:bg-gray-100 dark:hover:bg-gray-800 border-l-4 border-transparent'
        }
      `}
    >
      <div className="flex items-start gap-3">
        {/* Selection checkbox */}
        {showSelect && (
          <div className="mt-0.5 flex-shrink-0" onClick={handleCheckboxClick}>
            <Checkbox
              checked={isSelected}
              onChange={handleSelectChange}
            />
          </div>
        )}

        <div
          className={`
            mt-0.5 w-6 h-6 rounded flex items-center justify-center flex-shrink-0
            ${isActive
              ? 'bg-primary/20 text-primary'
              : 'bg-gray-200 dark:bg-gray-700 text-gray-500 dark:text-gray-400'
            }
          `}
        >
          <MessageSquare size={14} />
        </div>

        <div className="flex-1 min-w-0">
          {/* Title */}
          <div
            className={`
              font-medium text-sm truncate
              ${isActive
                ? 'text-primary'
                : 'text-gray-900 dark:text-white'
              }
            `}
          >
            {title}
          </div>

          {/* Preview */}
          <div className="text-xs text-gray-500 dark:text-gray-400 truncate mt-0.5">
            {preview}
          </div>

          {/* Meta */}
          <div className="flex items-center gap-2 mt-1 text-xs text-gray-400 dark:text-gray-500">
            <span>{formatRelativeDate(session.updated_at)}</span>
            {session.message_count > 0 && (
              <>
                <span>-</span>
                <span>{session.message_count} messages</span>
              </>
            )}
          </div>
        </div>

        {/* Delete button */}
        {showDelete && (
          <button
            onClick={handleDelete}
            className="mt-0.5 p-1.5 rounded-lg text-gray-400 hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors flex-shrink-0"
            title="Delete session"
          >
            <Trash2 size={14} />
          </button>
        )}
      </div>
    </div>
  );
}
