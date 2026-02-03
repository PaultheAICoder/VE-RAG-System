import { useState, useEffect, useCallback } from 'react';
import {
  RefreshCw,
  Pause,
  Play,
  Trash2,
  Loader2,
  Clock,
  CheckCircle,
  XCircle,
  AlertTriangle,
} from 'lucide-react';
import { Card, Button, Badge, Alert } from '../../ui';
import { ConfirmModal } from './ConfirmModal';
import {
  getWarmingJobs,
  pauseWarmingJob,
  resumeWarmingJob,
  deleteWarmingJob,
} from '../../../api/cache';
import { useCacheWarmingStore } from '../../../stores/cacheWarmingStore';
import type { WarmingJobStatus } from '../../../types';

interface WarmingQueueCardProps {
  onJobDelete?: () => void;
}

const STATUS_CONFIG: Record<
  WarmingJobStatus,
  { label: string; variant: 'default' | 'primary' | 'success' | 'warning' | 'danger'; icon: React.ReactNode }
> = {
  pending: { label: 'Pending', variant: 'default', icon: <Clock size={14} /> },
  running: { label: 'Running', variant: 'primary', icon: <Loader2 size={14} className="animate-spin" /> },
  paused: { label: 'Paused', variant: 'warning', icon: <Pause size={14} /> },
  completed: { label: 'Completed', variant: 'success', icon: <CheckCircle size={14} /> },
  failed: { label: 'Failed', variant: 'danger', icon: <XCircle size={14} /> },
};

/**
 * Format a date string as relative time (e.g., "2 mins ago").
 */
function formatRelativeTime(dateStr: string | null): string {
  if (!dateStr) return '-';
  const date = new Date(dateStr);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffSecs = Math.floor(diffMs / 1000);
  const diffMins = Math.floor(diffSecs / 60);
  const diffHours = Math.floor(diffMins / 60);
  const diffDays = Math.floor(diffHours / 24);

  if (diffSecs < 60) return 'just now';
  if (diffMins < 60) return `${diffMins} min${diffMins === 1 ? '' : 's'} ago`;
  if (diffHours < 24) return `${diffHours} hour${diffHours === 1 ? '' : 's'} ago`;
  return `${diffDays} day${diffDays === 1 ? '' : 's'} ago`;
}

export function WarmingQueueCard({ onJobDelete }: WarmingQueueCardProps) {
  // Local state for UI
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [jobToDelete, setJobToDelete] = useState<string | null>(null);

  // Store state
  const {
    queueJobs,
    queueLoading,
    queueError,
    setQueueJobs,
    setQueueLoading,
    setQueueError,
    updateQueueJob,
    removeQueueJob,
  } = useCacheWarmingStore();

  // Fetch queue jobs
  const fetchQueue = useCallback(async () => {
    setQueueLoading(true);
    setQueueError(null);
    try {
      const response = await getWarmingJobs();
      setQueueJobs(response.jobs);
    } catch (err) {
      setQueueError(err instanceof Error ? err.message : 'Failed to fetch queue');
    } finally {
      setQueueLoading(false);
    }
  }, [setQueueJobs, setQueueLoading, setQueueError]);

  // Fetch queue on mount
  useEffect(() => {
    fetchQueue();
  }, [fetchQueue]);

  // Auto-refresh every 3 seconds to catch new jobs
  useEffect(() => {
    const interval = setInterval(fetchQueue, 3000);
    return () => clearInterval(interval);
  }, [fetchQueue]);

  // Handle pause
  const handlePause = async (jobId: string) => {
    setActionLoading(jobId);
    try {
      const updated = await pauseWarmingJob(jobId);
      updateQueueJob(updated);
    } catch (err) {
      setQueueError(err instanceof Error ? err.message : 'Failed to pause job');
    } finally {
      setActionLoading(null);
    }
  };

  // Handle resume
  const handleResume = async (jobId: string) => {
    setActionLoading(jobId);
    try {
      const updated = await resumeWarmingJob(jobId);
      updateQueueJob(updated);
    } catch (err) {
      setQueueError(err instanceof Error ? err.message : 'Failed to resume job');
    } finally {
      setActionLoading(null);
    }
  };

  // Handle delete confirmation
  const handleDeleteClick = (jobId: string) => {
    setJobToDelete(jobId);
    setShowDeleteConfirm(true);
  };

  // Handle delete
  const handleDelete = async () => {
    if (!jobToDelete) return;
    setActionLoading(jobToDelete);
    setShowDeleteConfirm(false);
    try {
      await deleteWarmingJob(jobToDelete);
      removeQueueJob(jobToDelete);
      onJobDelete?.();
    } catch (err) {
      setQueueError(err instanceof Error ? err.message : 'Failed to delete job');
    } finally {
      setActionLoading(null);
      setJobToDelete(null);
    }
  };

  return (
    <Card>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
          Warming Queue
        </h2>
        <Button
          variant="secondary"
          size="sm"
          icon={RefreshCw}
          onClick={fetchQueue}
          disabled={queueLoading}
        >
          {queueLoading ? 'Refreshing...' : 'Refresh'}
        </Button>
      </div>

      {queueError && (
        <Alert variant="danger" onClose={() => setQueueError(null)}>
          {queueError}
        </Alert>
      )}

      {queueJobs.length === 0 && !queueLoading && (
        <div className="text-center py-8">
          <AlertTriangle size={32} className="mx-auto text-gray-400 mb-2" />
          <p className="text-gray-500 dark:text-gray-400">No warming jobs found.</p>
          <p className="text-sm text-gray-400 dark:text-gray-500">
            Upload a file in the Cache Warming section to start a job.
          </p>
        </div>
      )}

      {queueJobs.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200 dark:border-gray-700">
                <th className="text-left py-2 px-3 font-medium text-gray-500 dark:text-gray-400">
                  Source
                </th>
                <th className="text-left py-2 px-3 font-medium text-gray-500 dark:text-gray-400">
                  Status
                </th>
                <th className="text-left py-2 px-3 font-medium text-gray-500 dark:text-gray-400">
                  Progress
                </th>
                <th className="text-left py-2 px-3 font-medium text-gray-500 dark:text-gray-400">
                  Created
                </th>
                <th className="text-right py-2 px-3 font-medium text-gray-500 dark:text-gray-400">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody>
              {queueJobs.map((job) => {
                const statusConfig = STATUS_CONFIG[job.status];
                const progressPercent = job.total > 0
                  ? Math.round((job.processed / job.total) * 100)
                  : 0;
                const isJobLoading = actionLoading === job.id;

                return (
                  <tr
                    key={job.id}
                    className="border-b border-gray-100 dark:border-gray-800 last:border-0"
                  >
                    {/* Source File */}
                    <td className="py-3 px-3">
                      <span className="text-gray-900 dark:text-white">
                        {job.source_file || 'Manual Entry'}
                      </span>
                    </td>

                    {/* Status */}
                    <td className="py-3 px-3">
                      <Badge variant={statusConfig.variant}>
                        <span className="flex items-center gap-1.5">
                          {statusConfig.icon}
                          {statusConfig.label}
                        </span>
                      </Badge>
                    </td>

                    {/* Progress */}
                    <td className="py-3 px-3">
                      <div className="min-w-[120px]">
                        <div className="flex justify-between text-xs mb-1">
                          <span className="text-gray-600 dark:text-gray-400">
                            {job.processed}/{job.total}
                          </span>
                          <span className="font-medium text-gray-900 dark:text-white">
                            {progressPercent}%
                          </span>
                        </div>
                        <div className="h-1.5 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full transition-all duration-300 ${
                              job.status === 'failed'
                                ? 'bg-red-500'
                                : job.status === 'completed'
                                ? 'bg-green-500'
                                : 'bg-primary'
                            }`}
                            style={{ width: `${progressPercent}%` }}
                          />
                        </div>
                        {job.failed_count > 0 && (
                          <span className="text-xs text-red-500 mt-0.5 block">
                            {job.failed_count} failed
                          </span>
                        )}
                      </div>
                    </td>

                    {/* Created */}
                    <td className="py-3 px-3">
                      <span className="text-gray-500 dark:text-gray-400">
                        {formatRelativeTime(job.created_at)}
                      </span>
                    </td>

                    {/* Actions */}
                    <td className="py-3 px-3 text-right">
                      <div className="flex justify-end gap-1">
                        {/* Pause button - only for running jobs */}
                        {job.status === 'running' && (
                          <Button
                            variant="secondary"
                            size="sm"
                            icon={Pause}
                            onClick={() => handlePause(job.id)}
                            disabled={isJobLoading}
                            title="Pause"
                          />
                        )}

                        {/* Resume button - only for paused jobs */}
                        {job.status === 'paused' && (
                          <Button
                            variant="primary"
                            size="sm"
                            icon={Play}
                            onClick={() => handleResume(job.id)}
                            disabled={isJobLoading}
                            title="Resume"
                          />
                        )}

                        {/* Delete button - for all jobs */}
                        <Button
                          variant="danger"
                          size="sm"
                          icon={Trash2}
                          onClick={() => handleDeleteClick(job.id)}
                          disabled={isJobLoading}
                          title="Delete"
                        />
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Delete Confirmation Modal */}
      <ConfirmModal
        isOpen={showDeleteConfirm}
        onClose={() => {
          setShowDeleteConfirm(false);
          setJobToDelete(null);
        }}
        onConfirm={handleDelete}
        title="Delete Warming Job"
        message="Are you sure you want to delete this warming job? This action cannot be undone."
        confirmLabel="Delete"
        variant="danger"
        isLoading={!!actionLoading && actionLoading === jobToDelete}
      />
    </Card>
  );
}
