import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

interface MetricScoreCardProps {
  label: string;
  value: number | null;
  previousValue?: number | null;
  size?: 'sm' | 'md';
}

function getScoreColor(value: number): string {
  if (value >= 0.7) return 'text-green-600 dark:text-green-400';
  if (value >= 0.4) return 'text-amber-600 dark:text-amber-400';
  return 'text-red-600 dark:text-red-400';
}

function getScoreBg(value: number): string {
  if (value >= 0.7) return 'bg-green-50 dark:bg-green-900/20';
  if (value >= 0.4) return 'bg-amber-50 dark:bg-amber-900/20';
  return 'bg-red-50 dark:bg-red-900/20';
}

export function MetricScoreCard({ label, value, previousValue, size = 'md' }: MetricScoreCardProps) {
  const displayValue = value !== null ? (value * 100).toFixed(1) + '%' : 'N/A';
  const isSmall = size === 'sm';

  let TrendIcon = Minus;
  let trendColor = 'text-gray-400';
  if (value !== null && previousValue !== null && previousValue !== undefined) {
    if (value > previousValue + 0.01) {
      TrendIcon = TrendingUp;
      trendColor = 'text-green-500';
    } else if (value < previousValue - 0.01) {
      TrendIcon = TrendingDown;
      trendColor = 'text-red-500';
    }
  }

  return (
    <div
      className={`rounded-lg p-3 ${value !== null ? getScoreBg(value) : 'bg-gray-50 dark:bg-gray-800'}`}
    >
      <div className="flex items-center justify-between">
        <span className={`text-xs font-medium text-gray-500 dark:text-gray-400 ${isSmall ? 'text-[10px]' : ''}`}>
          {label}
        </span>
        {previousValue !== undefined && (
          <TrendIcon className={`w-3 h-3 ${trendColor}`} />
        )}
      </div>
      <div
        className={`font-bold mt-1 ${isSmall ? 'text-lg' : 'text-2xl'} ${value !== null ? getScoreColor(value) : 'text-gray-400 dark:text-gray-500'}`}
      >
        {displayValue}
      </div>
    </div>
  );
}
