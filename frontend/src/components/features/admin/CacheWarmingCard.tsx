import React, { Fragment, useState, useEffect, useCallback, useRef, useMemo } from 'react';
import {
  Flame,
  CheckCircle,
  AlertCircle,
  Upload,
  FileText,
  RotateCcw,
  X,
  Pause,
  Play,
  Trash2,
  Loader2,
  Clock,
  XCircle,
  AlertTriangle,
  StopCircle,
  ChevronDown,
  ChevronRight,
} from 'lucide-react';
import { Card, Button, Badge, Alert } from '../../ui';
import { ConfirmModal } from './ConfirmModal';
import { useAuthStore } from '../../../stores/authStore';
import { useCacheWarmingStore } from '../../../stores/cacheWarmingStore';
import {
  warmCacheFromFile,
  getWarmProgressUrl,
  getWarmStatus,
  getWarmingJobs,
  pauseWarmingJob,
  resumeWarmingJob,
  deleteWarmingJob,
  cancelWarmingJob,
  bulkDeleteWarmingJobs,
  getBatchQueries,
  retryBatch,
  retryQuery,
} from '../../../api/cache';
import type { WarmingJobStatus, WarmingQueryStatus } from '../../../types';

interface CacheWarmingCardProps {
  onWarm: (queries: string[]) => Promise<void>;
  onWarmingComplete?: () => void;
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
  completed_with_errors: { label: 'Completed with Errors', variant: 'warning', icon: <AlertTriangle size={14} /> },
  failed: { label: 'Failed', variant: 'danger', icon: <XCircle size={14} /> },
  cancelled: { label: 'Cancelled', variant: 'danger', icon: <StopCircle size={14} /> },
};

const QUERY_STATUS_CONFIG: Record<
  WarmingQueryStatus,
  { label: string; variant: 'default' | 'primary' | 'success' | 'warning' | 'danger'; icon: React.ReactNode }
> = {
  pending: { label: 'Pending', variant: 'default', icon: <Clock size={14} /> },
  processing: { label: 'Processing', variant: 'primary', icon: <Loader2 size={14} className="animate-spin" /> },
  completed: { label: 'Completed', variant: 'success', icon: <CheckCircle size={14} /> },
  failed: { label: 'Failed', variant: 'danger', icon: <XCircle size={14} /> },
  skipped: { label: 'Skipped', variant: 'default', icon: <StopCircle size={14} /> },
};

// Transitional states for jobs that are in process of cancelling/pausing
const TRANSITIONAL_STATES = {
  cancelling: { label: 'Cancelling...', variant: 'warning' as const, icon: <Loader2 size={14} className="animate-spin" /> },
  pausing: { label: 'Pausing...', variant: 'warning' as const, icon: <Loader2 size={14} className="animate-spin" /> },
};

/**
 * Get the display status config for a job, accounting for transitional states.
 */
function getJobStatusConfig(job: { status: WarmingJobStatus; is_cancel_requested?: boolean; is_paused?: boolean }) {
  // Check for transitional states
  if (job.status === 'running' && job.is_cancel_requested) {
    return TRANSITIONAL_STATES.cancelling;
  }
  if (job.status === 'running' && job.is_paused) {
    return TRANSITIONAL_STATES.pausing;
  }
  return STATUS_CONFIG[job.status];
}

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

type TabType = 'manual' | 'file';

export function CacheWarmingCard({
  onWarm,
  onWarmingComplete,
  onJobDelete,
}: CacheWarmingCardProps) {
  const [activeTab, setActiveTab] = useState<TabType>('manual');
  const [queryText, setQueryText] = useState('');
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);
  const [showFailedDetails, setShowFailedDetails] = useState(false);
  const eventSourceRef = useRef<EventSource | null>(null);
  const completedRef = useRef<boolean>(false);
  const jobStartTimeRef = useRef<number | null>(null);

  // Queue UI state
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [jobToDelete, setJobToDelete] = useState<string | null>(null);
  const [isBulkDelete, setIsBulkDelete] = useState(false);
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [expandedBatchIds, setExpandedBatchIds] = useState<Set<string>>(new Set());

  const {
    status,
    queriesCount,
    lastWarmingTime,
    error,
    activeJobId,
    processed,
    total,
    failedQueries,
    estimatedTimeRemaining,
    startWarming,
    setError,
    reset,
    startFileJob,
    updateProgress,
    completeFileJob,
    resetFileJob,
    // Queue state
    queueJobs,
    queueLoading,
    queueError,
    selectedJobIds,
    lastEventId,
    setQueueJobs,
    setQueueLoading,
    setQueueError,
    updateQueueJob,
    removeQueueJob,
    toggleJobSelection,
    selectAllJobs,
    clearSelection,
    setLastEventId,
    // Query expansion state
    expandedBatchQueries,
    expandedBatchLoading,
    setExpandedBatchQueries,
    setExpandedBatchLoading,
    clearExpandedBatch,
  } = useCacheWarmingStore();

  const isWarming = status === 'warming';

  const queries = queryText
    .split('\n')
    .map((q) => q.trim())
    .filter((q) => q.length > 0);

  // Cleanup EventSource on unmount
  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);

  // Check for orphaned jobs on mount (server restarted while job was running)
  useEffect(() => {
    const checkOrphanedJob = async () => {
      if (!activeJobId || status !== 'warming') return;

      try {
        const jobStatus = await getWarmStatus(activeJobId);

        if (jobStatus.status === 'running') {
          // Job still running, reconnect to SSE
          connectToSSE(activeJobId);
        } else if (jobStatus.status === 'completed') {
          // Job completed while we were away
          completeFileJob();
        } else if (jobStatus.status === 'failed') {
          // Job failed
          setError('Warming job failed on server');
          resetFileJob();
        }
      } catch {
        // Job not found (404) or server error - reset UI
        console.warn('Orphaned warming job detected, resetting UI:', activeJobId);
        resetFileJob();
      }
    };

    checkOrphanedJob();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Only run on mount - intentionally checking stored state once

  // Call onWarmingComplete when warming finishes
  useEffect(() => {
    if (status === 'completed' && onWarmingComplete) {
      onWarmingComplete();
    }
  }, [status, onWarmingComplete]);

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

  // SSE connection with lastEventId support for reconnection
  const connectToSSE = useCallback((jobId: string) => {
    const url = getWarmProgressUrl(jobId, lastEventId);
    if (!url) {
      setError('Authentication token missing. Please log in again.');
      return;
    }
    const eventSource = new EventSource(url);
    eventSourceRef.current = eventSource;
    completedRef.current = false;
    jobStartTimeRef.current = Date.now();

    eventSource.addEventListener('progress', (e) => {
      const data = JSON.parse(e.data);
      updateProgress(data);
      // Track lastEventId for reconnection
      if ((e as MessageEvent).lastEventId) {
        setLastEventId((e as MessageEvent).lastEventId);
      }
    });

    eventSource.addEventListener('paused', (e) => {
      const data = JSON.parse(e.data);
      updateProgress({ ...data, percent: 0, batch_status: 'paused' });
      if ((e as MessageEvent).lastEventId) {
        setLastEventId((e as MessageEvent).lastEventId);
      }
      fetchQueue(); // Refresh queue to show paused state
    });

    eventSource.addEventListener('complete', (e) => {
      console.log('[SSE] Complete event received:', e.data);
      completedRef.current = true;
      completeFileJob();
      setLastEventId(null); // Clear on completion
      eventSource.close();
      eventSourceRef.current = null;
      fetchQueue(); // Refresh queue after completion
    });

    eventSource.onerror = () => {
      // Delay error handling to allow complete event to process first
      // (SSE fires onerror when connection closes, even after normal completion)
      setTimeout(() => {
        const currentStatus = useCacheWarmingStore.getState().status;
        console.log('[SSE] Error handler - completedRef:', completedRef.current, 'status:', currentStatus);

        // Ignore if job already completed successfully
        if (completedRef.current || currentStatus === 'completed') {
          console.log('[SSE] Ignoring error - job completed');
          return;
        }

        // Check if token is still valid
        const token = useAuthStore.getState().token;
        if (!token) {
          setError('Session expired. Please log in again.');
        } else {
          setError('Connection to server lost. Please try again.');
        }
      }, 100);

      eventSource.close();
      eventSourceRef.current = null;
    };
  }, [updateProgress, completeFileJob, setError, lastEventId, setLastEventId, fetchQueue]);

  // Calculate queries per second
  const queriesPerSecond = useMemo(() => {
    if (!activeJobId || processed === 0 || !jobStartTimeRef.current) return null;
    const elapsed = (Date.now() - jobStartTimeRef.current) / 1000;
    return elapsed > 0 ? (processed / elapsed).toFixed(1) : null;
  }, [activeJobId, processed]);

  // Fetch queue on mount
  useEffect(() => {
    fetchQueue();
  }, [fetchQueue]);

  // Auto-refresh when jobs are active (running or pending)
  useEffect(() => {
    const hasActive = queueJobs.some((j) => j.status === 'running' || j.status === 'pending');
    if (hasActive) {
      const interval = setInterval(fetchQueue, 2000);
      return () => clearInterval(interval);
    }
  }, [queueJobs, fetchQueue]);

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

  // Handle cancel running job
  const handleCancel = async () => {
    setActionLoading('cancel');
    try {
      await cancelWarmingJob();
      await fetchQueue();
    } catch (err) {
      setQueueError(err instanceof Error ? err.message : 'Failed to cancel job');
    } finally {
      setActionLoading(null);
    }
  };

  // Handle delete confirmation click
  const handleDeleteClick = (jobId: string) => {
    setJobToDelete(jobId);
    setIsBulkDelete(false);
    setShowDeleteConfirm(true);
  };

  // Handle bulk delete click
  const handleBulkDeleteClick = () => {
    setIsBulkDelete(true);
    setShowDeleteConfirm(true);
  };

  // Handle delete (single or bulk)
  const handleDelete = async () => {
    setShowDeleteConfirm(false);

    if (isBulkDelete) {
      // Bulk delete
      const idsToDelete = Array.from(selectedJobIds);
      setActionLoading('bulk');
      try {
        await bulkDeleteWarmingJobs(idsToDelete);
        idsToDelete.forEach((id) => removeQueueJob(id));
        clearSelection();
        onJobDelete?.();
      } catch (err) {
        setQueueError(err instanceof Error ? err.message : 'Failed to delete jobs');
      } finally {
        setActionLoading(null);
      }
    } else if (jobToDelete) {
      // Single delete
      setActionLoading(jobToDelete);
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
    }
  };

  // Handle select all toggle
  const handleSelectAll = () => {
    if (selectedJobIds.size === queueJobs.length) {
      clearSelection();
    } else {
      selectAllJobs();
    }
  };

  // Handle manual warming
  const handleManualWarm = async () => {
    if (queries.length === 0) return;

    startWarming(queries.length);

    try {
      await onWarm(queries);
      reset(); // Return to idle — queue row shows real-time status
      setQueryText('');
      await fetchQueue(); // Refresh queue to show new batch
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Warming failed';
      setError(message.includes('503') ? 'Redis unavailable -- please retry' : message);
    }
  };

  // Handle file selection
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    setFileError(null);

    if (!file) {
      setUploadedFile(null);
      return;
    }

    const ext = file.name.toLowerCase().split('.').pop();
    if (ext !== 'txt' && ext !== 'csv') {
      setFileError('Only .txt and .csv files are supported');
      setUploadedFile(null);
      return;
    }

    setUploadedFile(file);
  };

  // Handle file upload and warming
  const handleFileWarm = async () => {
    if (!uploadedFile) return;

    setFileError(null);

    try {
      const response = await warmCacheFromFile(uploadedFile);
      startFileJob(response.id, response.total_queries);
      connectToSSE(response.id);
      await fetchQueue(); // Refresh queue to show new batch
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to start warming';
      setFileError(message.includes('503') ? 'Redis unavailable -- please retry' : message);
    }
  };

  // Handle retry all failed queries in a batch
  const handleRetryBatch = async (batchId: string) => {
    setActionLoading(`retry-${batchId}`);
    try {
      await retryBatch(batchId);
      await fetchQueue();
      // Refresh expanded queries if this batch is expanded
      if (expandedBatchIds.has(batchId)) {
        const response = await getBatchQueries(batchId);
        setExpandedBatchQueries(batchId, response.queries);
      }
    } catch (err) {
      setQueueError(err instanceof Error ? err.message : 'Retry failed');
    } finally {
      setActionLoading(null);
    }
  };

  // Handle retry a single query
  const handleRetryQuery = async (batchId: string, queryId: string) => {
    setActionLoading(`retry-query-${queryId}`);
    try {
      await retryQuery(batchId, queryId);
      // Refresh the expanded query list
      const response = await getBatchQueries(batchId);
      setExpandedBatchQueries(batchId, response.queries);
      await fetchQueue(); // Also refresh batch-level counts
    } catch (err) {
      setQueueError(err instanceof Error ? err.message : 'Query retry failed');
    } finally {
      setActionLoading(null);
    }
  };

  // Toggle batch row expansion to show individual queries
  const toggleBatchExpand = async (batchId: string) => {
    const newSet = new Set(expandedBatchIds);
    if (newSet.has(batchId)) {
      newSet.delete(batchId);
      clearExpandedBatch(batchId);
    } else {
      newSet.add(batchId);
      // Lazy-load queries
      setExpandedBatchLoading(batchId, true);
      try {
        const response = await getBatchQueries(batchId);
        setExpandedBatchQueries(batchId, response.queries);
      } catch (err) {
        setQueueError(err instanceof Error ? err.message : 'Failed to load queries');
      } finally {
        setExpandedBatchLoading(batchId, false);
      }
    }
    setExpandedBatchIds(newSet);
  };

  // Handle retry failed (legacy flow -- now delegates to batch retry)
  const handleRetryFailed = async () => {
    if (!activeJobId && failedQueries.length === 0) return;

    if (activeJobId) {
      await handleRetryBatch(activeJobId);
    }
    resetFileJob();
    setFileError(null);
  };

  // Handle reset
  const handleReset = () => {
    completedRef.current = false;
    reset();
    resetFileJob();
    setUploadedFile(null);
    setFileError(null);
    setShowFailedDetails(false);
  };

  // Format time remaining
  const formatTimeRemaining = (seconds: number | null): string => {
    if (seconds === null) return '';
    if (seconds < 60) return `${seconds}s remaining`;
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}m ${secs}s remaining`;
  };

  // Format last warming time
  const formatLastWarmingTime = (isoString: string | null): string => {
    if (!isoString) return '';
    try {
      return new Date(isoString).toLocaleString('en-US', {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      });
    } catch {
      return '';
    }
  };

  return (
    <Card>
      <div className="flex items-center gap-3 mb-4">
        <div className="w-10 h-10 rounded-xl flex items-center justify-center bg-amber-100 dark:bg-amber-900/30 text-amber-600 dark:text-amber-400">
          <Flame size={20} />
        </div>
        <h3 className="font-semibold text-gray-900 dark:text-white">Cache Warming</h3>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-gray-200 dark:border-gray-700 mb-4">
        <button
          onClick={() => setActiveTab('manual')}
          disabled={isWarming}
          className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
            activeTab === 'manual'
              ? 'border-primary text-primary'
              : 'border-transparent text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
          } ${isWarming ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          Manual Entry
        </button>
        <button
          onClick={() => setActiveTab('file')}
          disabled={isWarming}
          className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
            activeTab === 'file'
              ? 'border-primary text-primary'
              : 'border-transparent text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
          } ${isWarming ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          File Upload
        </button>
      </div>

      {/* Status Messages — only shown for faults (queue row is the primary indicator) */}
      {status === 'completed' && !isWarming && failedQueries.length > 0 && (
        <div className="flex items-center gap-2 p-3 mb-4 rounded-lg bg-amber-50 dark:bg-amber-900/20 text-amber-700 dark:text-amber-400">
          <AlertTriangle size={18} />
          <span className="text-sm flex-1">
            Warming completed with issues: {total > 0 ? `${total - failedQueries.length}/${total}` : `${queriesCount}/${queriesCount}`} queries succeeded
            {lastWarmingTime && ` at ${formatLastWarmingTime(lastWarmingTime)}`}
            {` (${failedQueries.length} failed)`}
          </span>
          <button
            onClick={handleReset}
            className="text-xs text-amber-600 dark:text-amber-400 hover:underline"
          >
            Dismiss
          </button>
        </div>
      )}

      {(status === 'error' || fileError) && (
        <div className="flex items-center gap-2 p-3 mb-4 rounded-lg bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-400">
          <AlertCircle size={18} />
          <span className="text-sm flex-1">{fileError || error || 'An error occurred'}</span>
          <button
            onClick={handleReset}
            className="text-xs text-red-600 dark:text-red-400 hover:underline"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Progress Display */}
      {isWarming && activeJobId && (
        <div className="mb-4 p-4 rounded-lg bg-amber-50 dark:bg-amber-900/20">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-amber-700 dark:text-amber-400">
              Warming Progress
            </span>
            <span className="text-sm text-amber-600 dark:text-amber-500">
              {processed}/{total} queries
            </span>
          </div>
          <div className="w-full bg-amber-200 dark:bg-amber-800 rounded-full h-2 mb-2">
            <div
              className="bg-amber-500 h-2 rounded-full transition-all duration-300"
              style={{ width: `${total > 0 ? (processed / total) * 100 : 0}%` }}
            />
          </div>
          {estimatedTimeRemaining !== null && (
            <p className="text-xs text-amber-600 dark:text-amber-500">
              {formatTimeRemaining(estimatedTimeRemaining)}
            </p>
          )}
        </div>
      )}

      {/* Failed Queries */}
      {failedQueries.length > 0 && status === 'completed' && (
        <div className="mb-4">
          <button
            onClick={() => setShowFailedDetails(!showFailedDetails)}
            className="text-sm text-red-600 dark:text-red-400 hover:underline flex items-center gap-1"
          >
            {showFailedDetails ? 'Hide' : 'Show'} {failedQueries.length} failed queries
          </button>
          {showFailedDetails && (
            <div className="mt-2 p-3 rounded-lg bg-red-50 dark:bg-red-900/20 max-h-32 overflow-y-auto">
              {failedQueries.map((q, i) => (
                <p key={i} className="text-xs text-red-700 dark:text-red-400 truncate">
                  {q}
                </p>
              ))}
            </div>
          )}
          <Button
            variant="outline"
            size="sm"
            icon={RotateCcw}
            onClick={handleRetryFailed}
            className="mt-2"
          >
            Retry Failed
          </Button>
        </div>
      )}

      {/* Manual Entry Tab */}
      {activeTab === 'manual' && (
        <div className="space-y-4">
          <p className="text-sm text-gray-500 dark:text-gray-400">
            Enter queries to pre-populate the cache, one per line.
          </p>
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
              {queries.length} {queries.length === 1 ? 'query' : 'queries'}
            </span>
            <Button
              icon={Flame}
              onClick={handleManualWarm}
              disabled={isWarming || queries.length === 0}
            >
              {isWarming ? 'Warming...' : 'Warm Cache'}
            </Button>
          </div>
        </div>
      )}

      {/* File Upload Tab */}
      {activeTab === 'file' && (
        <div className="space-y-4">
          <p className="text-sm text-gray-500 dark:text-gray-400">
            Upload a file with one query per line. Numbered queries (e.g., &quot;1. Question&quot;) are automatically cleaned.
          </p>

          <div className="border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg p-6 text-center">
            <input
              type="file"
              accept=".txt,.csv"
              onChange={handleFileSelect}
              disabled={isWarming}
              className="hidden"
              id="warm-file-input"
            />
            <label
              htmlFor="warm-file-input"
              className={`cursor-pointer ${isWarming ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
              {uploadedFile ? (
                <div className="flex items-center justify-center gap-2">
                  <FileText size={20} className="text-primary" />
                  <span className="text-sm font-medium text-gray-900 dark:text-white">
                    {uploadedFile.name}
                  </span>
                  <button
                    onClick={(e) => {
                      e.preventDefault();
                      setUploadedFile(null);
                    }}
                    className="text-gray-400 hover:text-gray-600"
                  >
                    <X size={16} />
                  </button>
                </div>
              ) : (
                <div className="flex flex-col items-center gap-2">
                  <Upload size={24} className="text-gray-400" />
                  <span className="text-sm text-gray-500 dark:text-gray-400">
                    Click to select a file
                  </span>
                </div>
              )}
            </label>
          </div>

          <div className="flex justify-end">
            <Button
              icon={Flame}
              onClick={handleFileWarm}
              disabled={isWarming || !uploadedFile}
            >
              {isWarming ? 'Warming...' : 'Start Warming'}
            </Button>
          </div>
        </div>
      )}

      {/* Warming Queue Section */}
      <div className="border-t border-gray-200 dark:border-gray-700 pt-4 mt-4">
        <div className="flex items-center justify-between mb-4">
          <h4 className="font-medium text-gray-900 dark:text-white">Warming Queue</h4>
          <div className="flex gap-2">
            {selectedJobIds.size > 0 && (
              <Button
                variant="danger"
                size="sm"
                icon={Trash2}
                onClick={handleBulkDeleteClick}
              >
                Delete ({selectedJobIds.size})
              </Button>
            )}
          </div>
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
              Upload a file above to start a job.
            </p>
          </div>
        )}

        {queueJobs.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left py-2 px-2 font-medium text-gray-500 dark:text-gray-400 w-8">
                    <input
                      type="checkbox"
                      checked={selectedJobIds.size === queueJobs.length && queueJobs.length > 0}
                      onChange={handleSelectAll}
                      className="rounded border-gray-300 text-primary focus:ring-primary"
                    />
                  </th>
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
                  // Use transitional state-aware status config
                  const statusConfig = getJobStatusConfig(job);
                  const jobProcessed = job.total_queries - job.pending_queries;
                  const progressPercent = job.total_queries > 0
                    ? Math.round((jobProcessed / job.total_queries) * 100)
                    : 0;
                  const isJobLoading = actionLoading === job.id;
                  // Check if job is in transitional state
                  const isCancelling = job.status === 'running' && job.is_cancel_requested;
                  const isPausing = job.status === 'running' && job.is_paused;
                  const isInTransition = isCancelling || isPausing;
                  const isExpanded = expandedBatchIds.has(job.id);

                  return (
                    <Fragment key={job.id}>
                    <tr
                      className="border-b border-gray-100 dark:border-gray-800 last:border-0"
                    >
                      {/* Checkbox */}
                      <td className="py-3 px-2">
                        <input
                          type="checkbox"
                          checked={selectedJobIds.has(job.id)}
                          onChange={() => toggleJobSelection(job.id)}
                          className="rounded border-gray-300 text-primary focus:ring-primary"
                        />
                      </td>

                      {/* Source - clickable to expand */}
                      <td className="py-3 px-3 cursor-pointer" onClick={() => toggleBatchExpand(job.id)}>
                        <span className="flex items-center gap-1">
                          {isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                          <span className="text-gray-900 dark:text-white">
                            {job.original_filename ?? (job.source_type === 'manual' ? 'Manual Entry' : job.source_type)}
                          </span>
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
                        {/* Tooltip for transitional states */}
                        {isCancelling && (
                          <span className="text-xs text-gray-500 block mt-1">
                            Waiting for current query to complete or timeout...
                          </span>
                        )}
                        {isPausing && (
                          <span className="text-xs text-gray-500 block mt-1">
                            Waiting for current query to complete...
                          </span>
                        )}
                      </td>

                      {/* Progress */}
                      <td className="py-3 px-3">
                        <div className="min-w-[120px]">
                          <div className="flex justify-between text-xs mb-1">
                            <span className="text-gray-600 dark:text-gray-400">
                              {jobProcessed}/{job.total_queries}
                            </span>
                            <span className="font-medium text-gray-900 dark:text-white">
                              {progressPercent}%
                            </span>
                          </div>
                          <div className="h-1.5 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                            <div
                              className={`h-full rounded-full transition-all duration-300 ${
                                job.status === 'failed' || job.status === 'cancelled'
                                  ? 'bg-red-500'
                                  : job.status === 'completed'
                                  ? 'bg-green-500'
                                  : job.status === 'completed_with_errors'
                                  ? 'bg-amber-500'
                                  : isInTransition
                                  ? 'bg-amber-500'
                                  : 'bg-primary'
                              }`}
                              style={{ width: `${progressPercent}%` }}
                            />
                          </div>
                          {job.status === 'running' && !isInTransition && queriesPerSecond && job.id === activeJobId && (
                            <span className="text-xs text-gray-500 mt-0.5 block">
                              {queriesPerSecond} q/s
                              {estimatedTimeRemaining && ` - ${formatTimeRemaining(estimatedTimeRemaining)}`}
                            </span>
                          )}
                          {job.status === 'completed_with_errors' && (
                            <span className="text-xs text-amber-600 mt-0.5 block">
                              {job.failed_queries} of {job.total_queries} queries failed
                            </span>
                          )}
                          {job.status === 'failed' && (
                            <span className="text-xs text-red-500 mt-0.5 block">
                              All queries failed
                            </span>
                          )}
                          {job.status !== 'completed_with_errors' && job.status !== 'failed' && job.failed_queries > 0 && (
                            <span className="text-xs text-red-500 mt-0.5 block">
                              {job.failed_queries} failed
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
                          {/* Retry button - for completed_with_errors or failed batches */}
                          {(job.status === 'completed_with_errors' || job.status === 'failed') && (
                            <Button
                              variant="warning"
                              size="sm"
                              icon={RotateCcw}
                              onClick={() => handleRetryBatch(job.id)}
                              disabled={actionLoading === `retry-${job.id}`}
                              title="Retry Failed Queries"
                            />
                          )}

                          {/* Cancel button - only for running jobs, disabled if already cancelling */}
                          {job.status === 'running' && !isPausing && (
                            <Button
                              variant="warning"
                              size="sm"
                              icon={StopCircle}
                              onClick={handleCancel}
                              disabled={actionLoading === 'cancel' || isCancelling}
                              title={isCancelling ? "Cancel in progress..." : "Cancel"}
                            />
                          )}

                          {/* Pause button - only for running jobs, disabled if already pausing/cancelling */}
                          {job.status === 'running' && !isCancelling && (
                            <Button
                              variant="secondary"
                              size="sm"
                              icon={Pause}
                              onClick={() => handlePause(job.id)}
                              disabled={isJobLoading || isPausing}
                              title={isPausing ? "Pause in progress..." : "Pause"}
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

                    {/* Expanded query detail row */}
                    {isExpanded && (
                      <tr>
                        <td colSpan={7} className="p-0">
                          <div className="bg-gray-50 dark:bg-gray-800/50 px-6 py-3 border-b border-gray-200 dark:border-gray-700">
                            {expandedBatchLoading[job.id] ? (
                              <div className="flex items-center gap-2 py-4 justify-center">
                                <Loader2 size={16} className="animate-spin text-gray-400" />
                                <span className="text-sm text-gray-500">Loading queries...</span>
                              </div>
                            ) : expandedBatchQueries[job.id] && expandedBatchQueries[job.id].length > 0 ? (
                              <table className="w-full text-xs">
                                <thead>
                                  <tr className="border-b border-gray-200 dark:border-gray-600">
                                    <th className="text-left py-1.5 px-2 font-medium text-gray-500 dark:text-gray-400">Query</th>
                                    <th className="text-left py-1.5 px-2 font-medium text-gray-500 dark:text-gray-400 w-28">Status</th>
                                    <th className="text-center py-1.5 px-2 font-medium text-gray-500 dark:text-gray-400 w-20">Confidence</th>
                                    <th className="text-left py-1.5 px-2 font-medium text-gray-500 dark:text-gray-400">Error</th>
                                    <th className="text-center py-1.5 px-2 font-medium text-gray-500 dark:text-gray-400 w-16">Retries</th>
                                    <th className="text-right py-1.5 px-2 font-medium text-gray-500 dark:text-gray-400 w-20">Actions</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {expandedBatchQueries[job.id].map((query) => {
                                    const qStatus = QUERY_STATUS_CONFIG[query.status];
                                    return (
                                      <tr key={query.id} className="border-b border-gray-100 dark:border-gray-700 last:border-0">
                                        <td className="py-1.5 px-2 text-gray-900 dark:text-white max-w-xs truncate" title={query.query_text}>
                                          {query.query_text}
                                        </td>
                                        <td className="py-1.5 px-2">
                                          <Badge variant={qStatus.variant}>
                                            <span className="flex items-center gap-1">
                                              {qStatus.icon}
                                              {qStatus.label}
                                            </span>
                                          </Badge>
                                        </td>
                                        <td className="py-1.5 px-2 text-center">
                                          {query.confidence_score != null ? (
                                            <span className={`inline-block px-1.5 py-0.5 rounded text-xs font-medium ${
                                              query.confidence_score >= 70
                                                ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
                                                : query.confidence_score >= 40
                                                  ? 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400'
                                                  : 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
                                            }`}>
                                              {query.confidence_score}%
                                            </span>
                                          ) : (
                                            <span className="text-gray-400">&mdash;</span>
                                          )}
                                        </td>
                                        <td className="py-1.5 px-2 text-red-500 max-w-xs truncate" title={query.error_message ?? undefined}>
                                          {query.error_message || '-'}
                                        </td>
                                        <td className="py-1.5 px-2 text-center text-gray-500">
                                          {query.retry_count}
                                        </td>
                                        <td className="py-1.5 px-2 text-right">
                                          {(query.status === 'failed' || query.status === 'skipped') && (
                                            <Button
                                              variant="outline"
                                              size="sm"
                                              icon={RotateCcw}
                                              onClick={() => handleRetryQuery(job.id, query.id)}
                                              disabled={actionLoading === `retry-query-${query.id}`}
                                              title="Retry Query"
                                            />
                                          )}
                                        </td>
                                      </tr>
                                    );
                                  })}
                                </tbody>
                              </table>
                            ) : (
                              <p className="text-sm text-gray-500 py-2 text-center">No queries found.</p>
                            )}
                          </div>
                        </td>
                      </tr>
                    )}
                    </Fragment>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Delete Confirmation Modal */}
      <ConfirmModal
        isOpen={showDeleteConfirm}
        onClose={() => {
          setShowDeleteConfirm(false);
          setJobToDelete(null);
          setIsBulkDelete(false);
        }}
        onConfirm={handleDelete}
        title={isBulkDelete ? 'Delete Warming Jobs' : 'Delete Warming Job'}
        message={
          isBulkDelete
            ? `Are you sure you want to delete ${selectedJobIds.size} warming job${selectedJobIds.size === 1 ? '' : 's'}? This action cannot be undone.`
            : 'Are you sure you want to delete this warming job? This action cannot be undone.'
        }
        confirmLabel="Delete"
        variant="danger"
        isLoading={actionLoading === 'bulk' || (!!actionLoading && actionLoading === jobToDelete)}
      />
    </Card>
  );
}
