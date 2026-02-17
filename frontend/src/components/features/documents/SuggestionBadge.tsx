import { Sparkles } from 'lucide-react';

interface SuggestionBadgeProps {
  count: number;
  onClick: () => void;
}

export function SuggestionBadge({ count, onClick }: SuggestionBadgeProps) {
  if (count <= 0) return null;

  return (
    <button
      onClick={(e) => {
        e.stopPropagation();
        onClick();
      }}
      className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400 hover:bg-amber-200 dark:hover:bg-amber-900/50 transition-colors cursor-pointer"
      title={`${count} pending tag suggestion${count !== 1 ? 's' : ''}`}
    >
      <Sparkles size={12} />
      +{count}
    </button>
  );
}
