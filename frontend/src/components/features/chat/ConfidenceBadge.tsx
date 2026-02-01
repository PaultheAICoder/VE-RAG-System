import { Clock } from 'lucide-react';
import type { ConfidenceInfo } from '../../../types';
import { Badge } from '../../ui';

interface ConfidenceBadgeProps {
  confidence: ConfidenceInfo;
  generationTimeMs?: number;
  showBreakdown?: boolean;
}

// Get color classes based on score
function getScoreColor(score: number): string {
  if (score >= 80) return 'text-green-600 dark:text-green-400';
  if (score >= 50) return 'text-amber-600 dark:text-amber-400';
  return 'text-red-600 dark:text-red-400';
}

function getScoreBg(score: number): string {
  if (score >= 80) return 'bg-green-500';
  if (score >= 50) return 'bg-amber-500';
  return 'bg-red-500';
}

export function ConfidenceBadge({ confidence, generationTimeMs, showBreakdown = false }: ConfidenceBadgeProps) {
  const { overall } = confidence;

  // Color coding: green >= 80%, yellow 50-79%, red < 50%
  const variant = overall >= 80 ? 'success' : overall >= 50 ? 'warning' : 'danger';

  // Format time: show seconds if >= 1000ms, else show ms
  const formatTime = (ms: number) => {
    if (ms >= 1000) {
      return `${(ms / 1000).toFixed(1)}s`;
    }
    return `${Math.round(ms)}ms`;
  };

  if (showBreakdown) {
    return (
      <div className="inline-flex flex-col gap-2 p-3 rounded-lg bg-gray-50 dark:bg-gray-800/50 border border-gray-200 dark:border-gray-700">
        {/* Overall confidence header */}
        <div className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${getScoreBg(overall)}`} />
            <span className={`text-sm font-semibold ${getScoreColor(overall)}`}>
              {overall}% Confident
            </span>
          </div>
          {generationTimeMs && (
            <div className="flex items-center gap-1 text-xs text-gray-500 dark:text-gray-400">
              <Clock size={12} />
              <span>{formatTime(generationTimeMs)}</span>
            </div>
          )}
        </div>

        {/* Metrics grid */}
        <div className="grid grid-cols-3 gap-2 pt-1 border-t border-gray-200 dark:border-gray-600">
          <div className="flex flex-col">
            <span className="text-[10px] uppercase tracking-wide text-gray-400 dark:text-gray-500">Retrieval</span>
            <span className={`text-sm font-medium ${getScoreColor(confidence.retrieval)}`}>{confidence.retrieval}%</span>
          </div>
          <div className="flex flex-col">
            <span className="text-[10px] uppercase tracking-wide text-gray-400 dark:text-gray-500">Coverage</span>
            <span className={`text-sm font-medium ${getScoreColor(confidence.coverage)}`}>{confidence.coverage}%</span>
          </div>
          <div className="flex flex-col">
            <span className="text-[10px] uppercase tracking-wide text-gray-400 dark:text-gray-500">LLM</span>
            <span className={`text-sm font-medium ${getScoreColor(confidence.llm)}`}>{confidence.llm}%</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <Badge variant={variant}>
      {overall}% confident
    </Badge>
  );
}
