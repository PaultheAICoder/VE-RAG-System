import { Search } from 'lucide-react';
import { Card } from '../../ui';
import type { TopCachedQuery } from '../../../types';

interface CacheTopQueriesCardProps {
  queries: TopCachedQuery[];
  loading?: boolean;
}

function formatRelativeTime(isoString: string): string {
  try {
    const date = new Date(isoString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);

    if (diffMins < 1) return 'just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  } catch {
    return 'unknown';
  }
}

function truncateText(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text;
  return text.slice(0, maxLength - 3) + '...';
}

export function CacheTopQueriesCard({
  queries,
  loading = false,
}: CacheTopQueriesCardProps) {
  return (
    <Card>
      <div className="flex items-center gap-3 mb-4">
        <div className="w-10 h-10 rounded-xl flex items-center justify-center bg-primary/10 text-primary">
          <Search size={20} />
        </div>
        <h3 className="font-semibold text-gray-900 dark:text-white">Top Cached Queries</h3>
      </div>

      {loading ? (
        <div className="text-sm text-gray-500 dark:text-gray-400">Loading queries...</div>
      ) : queries.length === 0 ? (
        <div className="text-center py-8">
          <Search className="mx-auto h-12 w-12 text-gray-300 dark:text-gray-600" />
          <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">
            No cached queries yet
          </p>
          <p className="text-xs text-gray-400 dark:text-gray-500">
            Queries will appear here as users ask questions
          </p>
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left border-b border-gray-200 dark:border-gray-700">
                <th className="pb-2 font-medium text-gray-700 dark:text-gray-300">Query</th>
                <th className="pb-2 font-medium text-gray-700 dark:text-gray-300 text-center w-20">
                  Hits
                </th>
                <th className="pb-2 font-medium text-gray-700 dark:text-gray-300 text-right w-24">
                  Last Hit
                </th>
              </tr>
            </thead>
            <tbody>
              {queries.map((query, index) => (
                <tr
                  key={index}
                  className="border-b border-gray-100 dark:border-gray-700/50 last:border-0"
                >
                  <td
                    className="py-2 text-gray-900 dark:text-white"
                    title={query.query_text}
                  >
                    {truncateText(query.query_text, 60)}
                  </td>
                  <td className="py-2 text-center">
                    <span className="inline-flex items-center justify-center min-w-[2rem] px-2 py-0.5 rounded-full text-xs font-medium bg-primary/10 text-primary">
                      {query.hit_count}
                    </span>
                  </td>
                  <td className="py-2 text-right text-gray-500 dark:text-gray-400 text-xs">
                    {formatRelativeTime(query.last_hit)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}
