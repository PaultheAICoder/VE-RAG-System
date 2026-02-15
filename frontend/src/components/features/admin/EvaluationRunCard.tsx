import { Eye, XCircle, Trash2, Clock } from 'lucide-react';
import { Card, Badge, Button } from '../../ui';
import { MetricScoreCard } from './MetricScoreCard';
import type { EvaluationRun, EvaluationRunStatus } from '../../../types';

interface EvaluationRunCardProps {
  run: EvaluationRun;
  onViewSamples: (run: EvaluationRun) => void;
  onCancel: (run: EvaluationRun) => void;
  onDelete: (run: EvaluationRun) => void;
}

const STATUS_BADGE: Record<EvaluationRunStatus, 'default' | 'primary' | 'success' | 'warning' | 'danger'> = {
  pending: 'default',
  running: 'primary',
  completed: 'success',
  completed_with_errors: 'warning',
  failed: 'danger',
  cancelled: 'default',
};

function formatDate(iso: string | null): string {
  if (!iso) return '—';
  return new Date(iso).toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

function formatETA(seconds: number | null): string {
  if (seconds === null || seconds <= 0) return '';
  if (seconds < 60) return `~${Math.ceil(seconds)}s remaining`;
  return `~${Math.ceil(seconds / 60)}m remaining`;
}

export function EvaluationRunCard({ run, onViewSamples, onCancel, onDelete }: EvaluationRunCardProps) {
  const progress = run.total_samples > 0
    ? Math.round(((run.completed_samples + run.failed_samples) / run.total_samples) * 100)
    : 0;
  const isActive = run.status === 'pending' || run.status === 'running';
  const canDelete = !isActive;

  return (
    <Card className="hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between mb-3">
        <div>
          <h3 className="font-medium text-gray-900 dark:text-white">{run.name}</h3>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
            {run.model_used} &middot; {formatDate(run.created_at)}
          </p>
        </div>
        <Badge variant={STATUS_BADGE[run.status]}>{run.status.replace(/_/g, ' ')}</Badge>
      </div>

      {/* Progress bar for active runs */}
      {isActive && (
        <div className="mb-3">
          <div className="flex justify-between text-xs text-gray-500 mb-1">
            <span>{run.completed_samples + run.failed_samples} / {run.total_samples} samples</span>
            <span>{progress}%</span>
          </div>
          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
            <div
              className="bg-primary rounded-full h-2 transition-all"
              style={{ width: `${progress}%` }}
            />
          </div>
          {run.eta_seconds && (
            <p className="text-xs text-gray-400 mt-1 flex items-center gap-1">
              <Clock className="w-3 h-3" />
              {formatETA(run.eta_seconds)}
            </p>
          )}
        </div>
      )}

      {/* Metric scores for completed runs */}
      {run.status !== 'pending' && (
        <div className="grid grid-cols-2 gap-2 mb-3">
          <MetricScoreCard label="Faithfulness" value={run.avg_faithfulness} size="sm" />
          <MetricScoreCard label="Relevancy" value={run.avg_answer_relevancy} size="sm" />
        </div>
      )}

      {/* Stats row */}
      {run.failed_samples > 0 && (
        <p className="text-xs text-red-500 mb-2">
          {run.failed_samples} failed &middot; {run.invalid_score_count} invalid scores
        </p>
      )}

      {/* Actions */}
      <div className="flex gap-2 pt-2 border-t border-gray-100 dark:border-gray-700">
        <Button size="sm" variant="outline" icon={Eye} onClick={() => onViewSamples(run)}>
          Samples
        </Button>
        {isActive && !run.is_cancel_requested && (
          <Button size="sm" variant="warning" icon={XCircle} onClick={() => onCancel(run)}>
            Cancel
          </Button>
        )}
        {canDelete && (
          <Button size="sm" variant="danger" icon={Trash2} onClick={() => onDelete(run)}>
            Delete
          </Button>
        )}
      </div>
    </Card>
  );
}
