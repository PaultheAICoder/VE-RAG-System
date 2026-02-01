import { useState } from 'react';
import { Flame } from 'lucide-react';
import { Card, Button } from '../../ui';

interface CacheWarmingCardProps {
  onWarm: (queries: string[]) => Promise<void>;
  isWarming?: boolean;
}

export function CacheWarmingCard({
  onWarm,
  isWarming = false,
}: CacheWarmingCardProps) {
  const [queryText, setQueryText] = useState('');

  const queries = queryText
    .split('\n')
    .map((q) => q.trim())
    .filter((q) => q.length > 0);

  const handleWarm = async () => {
    if (queries.length === 0) return;
    await onWarm(queries);
    setQueryText(''); // Clear after successful warm
  };

  return (
    <Card>
      <div className="flex items-center gap-3 mb-4">
        <div className="w-10 h-10 rounded-xl flex items-center justify-center bg-amber-100 dark:bg-amber-900/30 text-amber-600 dark:text-amber-400">
          <Flame size={20} />
        </div>
        <h3 className="font-semibold text-gray-900 dark:text-white">Cache Warming</h3>
      </div>

      <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
        Pre-populate the cache with expected queries. Enter one query per line.
        These queries will be processed in the background.
      </p>

      <div className="space-y-4">
        <textarea
          value={queryText}
          onChange={(e) => setQueryText(e.target.value)}
          placeholder="What is our return policy?&#10;How do I request time off?&#10;What are the security requirements?"
          rows={5}
          disabled={isWarming}
          className="
            w-full px-3 py-2 rounded-lg
            bg-white dark:bg-gray-700
            border border-gray-300 dark:border-gray-600
            text-gray-900 dark:text-white
            placeholder-gray-400 dark:placeholder-gray-500
            focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary
            disabled:opacity-50 disabled:cursor-not-allowed
            resize-none
          "
        />

        <div className="flex items-center justify-between">
          <span className="text-xs text-gray-500 dark:text-gray-400">
            {queries.length} {queries.length === 1 ? 'query' : 'queries'} to warm
          </span>
          <Button
            icon={Flame}
            onClick={handleWarm}
            disabled={isWarming || queries.length === 0}
          >
            {isWarming ? 'Warming...' : 'Warm Cache'}
          </Button>
        </div>
      </div>
    </Card>
  );
}
