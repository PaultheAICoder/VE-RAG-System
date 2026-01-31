import { useState, useEffect, useCallback } from 'react';
import {
  RefreshCw,
  Pause,
  XCircle,
  SkipForward,
  RotateCcw,
  FastForward,
  AlertTriangle,
  CheckCircle,
  Clock,
  Loader2,
} from 'lucide-react';
import { Card, Button, Badge, Alert } from '../../ui';
import { ConfirmModal } from './ConfirmModal';
import { ReindexFailuresModal } from './ReindexFailuresModal';
import {
  getReindexStatus,
  pauseReindex,
  resumeReindex,
  abortReindex,
} from '../../../api/admin';
import type { ReindexJob, ResumeAction } from '../../../types';

const STATUS_CONFIG: Record<
  string,
  { label: string; variant: 'default' | 'primary' | 'success' | 'warning' | 'danger'; icon: React.ReactNode }
> = {
  pending: { label: 'Pending', variant: 'default', icon: <Clock size={14} /> },
  running: { label: 'Running', variant: 'primary', icon: <Loader2 size={14} className="animate-spin" /> },
  paused: { label: 'Paused', variant: 'warning', icon: <Pause size={14} /> },
  completed: { label: 'Completed', variant: 'success', icon: <CheckCircle size={14} /> },
  failed: { label: 'Failed', variant: 'danger', icon: <XCircle size={14} /> },
  aborted: { label: 'Aborted', variant: 'default', icon: <XCircle size={14} /> },
};

export function ReindexStatusCard() {
  const [job, setJob] = useState<ReindexJob | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [statusChecked, setStatusChecked] = useState(false);

  // Modal states
  const [showAbortConfirm, setShowAbortConfirm] = useState(false);
  const [showSkipAllConfirm, setShowSkipAllConfirm] = useState(false);
  const [showFailuresModal, setShowFailuresModal] = useState(false);

  const fetchStatus = useCallback(async () => {
    setLoading(true);
    setError(null);
    setStatusChecked(false);
    try {
      const data = await getReindexStatus();
      setJob(data);
      setStatusChecked(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch status');
    } finally {
      setLoading(false);
    }
  }, []);

  // Fetch status on mount
  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  // Auto-refresh when job is active (pending or running)
  useEffect(() => {
    if (job?.status === 'running' || job?.status === 'pending') {
      const interval = setInterval(fetchStatus, 2000);
      return () => clearInterval(interval);
    }
  }, [job?.status, fetchStatus]);

  const handlePause = async () => {
    setActionLoading('pause');
    try {
      const updated = await pauseReindex();
      setJob(updated);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to pause');
    } finally {
      setActionLoading(null);
    }
  };

  const handleResume = async (action: ResumeAction) => {
    if (action === 'skip_all') {
      setShowSkipAllConfirm(true);
      return;
    }
    setActionLoading(action);
    try {
      const updated = await resumeReindex(action);
      setJob(updated);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to resume');
    } finally {
      setActionLoading(null);
    }
  };

  const handleSkipAllConfirm = async () => {
    setShowSkipAllConfirm(false);
    setActionLoading('skip_all');
    try {
      const updated = await resumeReindex('skip_all');
      setJob(updated);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to enable skip all');
    } finally {
      setActionLoading(null);
    }
  };

  const handleAbort = async () => {
    setShowAbortConfirm(false);
    setActionLoading('abort');
    try {
      const updated = await abortReindex();
      setJob(updated);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to abort');
    } finally {
      setActionLoading(null);
    }
  };

  const statusConfig = job ? STATUS_CONFIG[job.status] || STATUS_CONFIG.pending : null;
  const isRetryDisabled = job ? job.retry_count >= job.max_retries : false;

  return (
    <Card>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white">Reindex Status</h2>
        <Button
          variant="secondary"
          size="sm"
          icon={RefreshCw}
          onClick={fetchStatus}
          disabled={loading}
        >
          {loading ? 'Checking...' : 'Check Status'}
        </Button>
      </div>

      {error && (
        <Alert variant="danger" onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {!job && !loading && !statusChecked && (
        <p className="text-gray-500 dark:text-gray-400 text-sm">
          No active reindex job. Click "Check Status" to refresh.
        </p>
      )}

      {!job && !loading && statusChecked && (
        <Alert variant="success" onClose={() => setStatusChecked(false)}>
          No active reindex job found. The system is idle.
        </Alert>
      )}

      {job && (
        <div className="space-y-4">
          {/* Status Badge */}
          <div className="flex items-center gap-3">
            <Badge variant={statusConfig?.variant || 'default'}>
              <span className="flex items-center gap-1.5">
                {statusConfig?.icon}
                {statusConfig?.label}
              </span>
            </Badge>
            {job.auto_skip_failures && (
              <Badge variant="warning">
                Auto-skip enabled
              </Badge>
            )}
          </div>

          {/* Progress Bar */}
          <div>
            <div className="flex justify-between text-sm mb-1">
              <span className="text-gray-600 dark:text-gray-400">Progress</span>
              <span className="font-medium text-gray-900 dark:text-white">
                {job.progress_percent.toFixed(1)}%
              </span>
            </div>
            <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-primary rounded-full transition-all duration-300"
                style={{ width: `${job.progress_percent}%` }}
              />
            </div>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-3 gap-4 text-center">
            <div>
              <div className="text-2xl font-bold text-gray-900 dark:text-white">
                {job.processed_documents}
              </div>
              <div className="text-xs text-gray-500 dark:text-gray-400">Processed</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-gray-900 dark:text-white">
                {job.total_documents}
              </div>
              <div className="text-xs text-gray-500 dark:text-gray-400">Total</div>
            </div>
            <div>
              <div className={`text-2xl font-bold ${job.failed_documents > 0 ? 'text-red-600' : 'text-gray-900 dark:text-white'}`}>
                {job.failed_documents}
              </div>
              <div className="text-xs text-gray-500 dark:text-gray-400">
                Failed
                {job.failed_documents > 0 && (
                  <button
                    onClick={() => setShowFailuresModal(true)}
                    className="ml-1 text-primary hover:underline"
                  >
                    [View]
                  </button>
                )}
              </div>
            </div>
          </div>

          {/* Paused Info */}
          {job.status === 'paused' && (
            <div className="p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg border border-yellow-200 dark:border-yellow-800">
              <div className="flex items-start gap-2">
                <AlertTriangle size={16} className="text-yellow-600 dark:text-yellow-400 mt-0.5" />
                <div className="text-sm">
                  <p className="font-medium text-yellow-800 dark:text-yellow-200">
                    Job Paused
                    {job.paused_reason === 'failure' && ' - Document Failed'}
                    {job.paused_reason === 'user_request' && ' - User Requested'}
                  </p>
                  {job.last_error && (
                    <p className="text-yellow-700 dark:text-yellow-300 mt-1 truncate">
                      {job.last_error}
                    </p>
                  )}
                  {job.retry_count > 0 && (
                    <p className="text-yellow-600 dark:text-yellow-400 mt-1">
                      Retry {job.retry_count}/{job.max_retries}
                    </p>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex flex-wrap gap-2 pt-2 border-t border-gray-200 dark:border-gray-700">
            {job.status === 'running' && (
              <>
                <Button
                  variant="secondary"
                  size="sm"
                  icon={Pause}
                  onClick={handlePause}
                  disabled={actionLoading !== null}
                >
                  {actionLoading === 'pause' ? 'Pausing...' : 'Pause'}
                </Button>
                <Button
                  variant="danger"
                  size="sm"
                  icon={XCircle}
                  onClick={() => setShowAbortConfirm(true)}
                  disabled={actionLoading !== null}
                >
                  Abort
                </Button>
              </>
            )}

            {job.status === 'paused' && (
              <>
                <Button
                  variant="secondary"
                  size="sm"
                  icon={SkipForward}
                  onClick={() => handleResume('skip')}
                  disabled={actionLoading !== null}
                >
                  {actionLoading === 'skip' ? 'Skipping...' : 'Skip'}
                </Button>
                <Button
                  variant="primary"
                  size="sm"
                  icon={RotateCcw}
                  onClick={() => handleResume('retry')}
                  disabled={actionLoading !== null || isRetryDisabled}
                  title={isRetryDisabled ? 'Max retries reached' : undefined}
                >
                  {actionLoading === 'retry' ? 'Retrying...' : 'Retry'}
                </Button>
                <Button
                  variant="warning"
                  size="sm"
                  icon={FastForward}
                  onClick={() => handleResume('skip_all')}
                  disabled={actionLoading !== null}
                >
                  {actionLoading === 'skip_all' ? 'Enabling...' : 'Skip All'}
                </Button>
                <Button
                  variant="danger"
                  size="sm"
                  icon={XCircle}
                  onClick={() => setShowAbortConfirm(true)}
                  disabled={actionLoading !== null}
                >
                  Abort
                </Button>
              </>
            )}
          </div>

          {isRetryDisabled && job.status === 'paused' && (
            <p className="text-xs text-amber-600 dark:text-amber-400">
              Max retries ({job.max_retries}) reached. Use Skip to continue.
            </p>
          )}
        </div>
      )}

      {/* Abort Confirmation Modal */}
      <ConfirmModal
        isOpen={showAbortConfirm}
        onClose={() => setShowAbortConfirm(false)}
        onConfirm={handleAbort}
        title="Abort Reindex"
        message="Are you sure you want to abort the reindex operation? This cannot be undone. Documents already processed will remain indexed."
        confirmLabel="Yes, Abort"
        variant="danger"
        isLoading={actionLoading === 'abort'}
      />

      {/* Skip All Confirmation Modal */}
      <ConfirmModal
        isOpen={showSkipAllConfirm}
        onClose={() => setShowSkipAllConfirm(false)}
        onConfirm={handleSkipAllConfirm}
        title="Enable Auto-Skip"
        message="This will automatically skip all future failures. Failed documents can be retried individually later. Continue?"
        confirmLabel="Yes, Skip All"
        variant="warning"
        isLoading={actionLoading === 'skip_all'}
      />

      {/* Failures Modal */}
      <ReindexFailuresModal
        isOpen={showFailuresModal}
        onClose={() => setShowFailuresModal(false)}
        jobId={job?.id}
        onRetrySuccess={fetchStatus}
      />
    </Card>
  );
}
