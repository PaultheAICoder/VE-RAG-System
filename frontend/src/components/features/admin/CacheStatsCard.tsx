import { Trash2, Database } from 'lucide-react';
import { Card, Button } from '../../ui';
import type { CacheStats } from '../../../types';

interface CacheStatsCardProps {
  stats: CacheStats | null;
  onClear: () => void;
  isClearing?: boolean;
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
}

function formatDate(isoString: string | null): string {
  if (!isoString) return 'N/A';
  try {
    return new Date(isoString).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return 'N/A';
  }
}

export function CacheStatsCard({
  stats,
  onClear,
  isClearing = false,
}: CacheStatsCardProps) {
  if (!stats) {
    return (
      <Card>
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 rounded-xl flex items-center justify-center bg-primary/10 text-primary">
            <Database size={20} />
          </div>
          <h3 className="font-semibold text-gray-900 dark:text-white">Cache Statistics</h3>
        </div>
        <p className="text-sm text-gray-500 dark:text-gray-400">Loading statistics...</p>
      </Card>
    );
  }

  const hitRate = stats.hit_rate.toFixed(1);
  const speedup = stats.avg_response_time_uncached_ms > 0
    ? (stats.avg_response_time_uncached_ms / Math.max(stats.avg_response_time_cached_ms, 1)).toFixed(1)
    : '0';
  const cachedBarWidth = stats.avg_response_time_uncached_ms > 0
    ? Math.min(100, (stats.avg_response_time_cached_ms / stats.avg_response_time_uncached_ms) * 100)
    : 0;

  return (
    <Card>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl flex items-center justify-center bg-primary/10 text-primary">
            <Database size={20} />
          </div>
          <h3 className="font-semibold text-gray-900 dark:text-white">Cache Statistics</h3>
        </div>
        <Button
          variant="danger"
          icon={Trash2}
          onClick={onClear}
          disabled={isClearing || stats.total_entries === 0}
        >
          {isClearing ? 'Clearing...' : 'Clear Cache'}
        </Button>
      </div>

      {/* Status Badge */}
      <div className="mb-4">
        <span
          className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
            stats.enabled
              ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400'
              : 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300'
          }`}
        >
          {stats.enabled ? 'Enabled' : 'Disabled'}
        </span>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="text-center p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
          <div className="text-2xl font-bold text-primary">{hitRate}%</div>
          <div className="text-xs text-gray-500 dark:text-gray-400">Hit Rate</div>
        </div>
        <div className="text-center p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
          <div className="text-2xl font-bold text-gray-900 dark:text-white">
            {stats.total_entries.toLocaleString()}
          </div>
          <div className="text-xs text-gray-500 dark:text-gray-400">
            Entries ({stats.memory_entries}m / {stats.sqlite_entries}db)
          </div>
        </div>
        <div className="text-center p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
          <div className="text-2xl font-bold text-green-600 dark:text-green-400">
            {stats.avg_response_time_cached_ms.toFixed(0)}ms
          </div>
          <div className="text-xs text-gray-500 dark:text-gray-400">Avg Cached</div>
        </div>
        <div className="text-center p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
          <div className="text-2xl font-bold text-amber-600 dark:text-amber-400">
            {stats.avg_response_time_uncached_ms.toFixed(0)}ms
          </div>
          <div className="text-xs text-gray-500 dark:text-gray-400">Avg Uncached</div>
        </div>
      </div>

      {/* Response Time Comparison Bar */}
      {stats.avg_response_time_uncached_ms > 0 && (
        <div className="mb-4">
          <div className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Response Time Comparison
          </div>
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <span className="text-xs text-gray-500 dark:text-gray-400 w-16">Cached</span>
              <div className="flex-1 h-4 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-green-500 rounded-full transition-all duration-300"
                  style={{ width: `${cachedBarWidth}%` }}
                />
              </div>
              <span className="text-xs text-gray-500 dark:text-gray-400 w-16 text-right">
                {stats.avg_response_time_cached_ms.toFixed(0)}ms
              </span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs text-gray-500 dark:text-gray-400 w-16">Uncached</span>
              <div className="flex-1 h-4 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                <div className="h-full bg-amber-500 rounded-full w-full" />
              </div>
              <span className="text-xs text-gray-500 dark:text-gray-400 w-16 text-right">
                {stats.avg_response_time_uncached_ms.toFixed(0)}ms
              </span>
            </div>
          </div>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-2 text-center">
            Cache provides <span className="font-semibold text-green-600 dark:text-green-400">{speedup}x</span> speedup
          </p>
        </div>
      )}

      {/* Additional Info */}
      <div className="text-xs text-gray-500 dark:text-gray-400 space-y-1 border-t border-gray-200 dark:border-gray-700 pt-4">
        <div className="flex justify-between">
          <span>Storage Size:</span>
          <span>{formatBytes(stats.storage_size_bytes)}</span>
        </div>
        <div className="flex justify-between">
          <span>Total Hits / Misses:</span>
          <span>{stats.hit_count.toLocaleString()} / {stats.miss_count.toLocaleString()}</span>
        </div>
        <div className="flex justify-between">
          <span>Oldest Entry:</span>
          <span>{formatDate(stats.oldest_entry)}</span>
        </div>
        <div className="flex justify-between">
          <span>Newest Entry:</span>
          <span>{formatDate(stats.newest_entry)}</span>
        </div>
      </div>
    </Card>
  );
}
