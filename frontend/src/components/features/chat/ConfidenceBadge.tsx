import type { ConfidenceInfo } from '../../../types';
import { Badge } from '../../ui';

interface ConfidenceBadgeProps {
  confidence: ConfidenceInfo;
  showBreakdown?: boolean;
}

export function ConfidenceBadge({ confidence, showBreakdown = false }: ConfidenceBadgeProps) {
  const { overall } = confidence;

  // Color coding: green >= 80%, yellow 50-79%, red < 50%
  const variant = overall >= 80 ? 'success' : overall >= 50 ? 'warning' : 'danger';

  if (showBreakdown) {
    return (
      <div className="flex flex-col gap-1">
        <div className="flex items-center gap-2">
          <Badge variant={variant}>{overall}% confident</Badge>
        </div>
        <div className="text-xs text-gray-500 dark:text-gray-400 space-y-0.5">
          <div>Retrieval: {confidence.retrieval}%</div>
          <div>Coverage: {confidence.coverage}%</div>
          <div>LLM: {confidence.llm}%</div>
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
