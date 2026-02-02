import { useState, useEffect, useCallback, useRef } from 'react';
import { Flame, CheckCircle, AlertCircle, Upload, FileText, RotateCcw, X } from 'lucide-react';
import { Card, Button } from '../../ui';
import { useCacheWarmingStore } from '../../../stores/cacheWarmingStore';
import { warmCacheFromFile, retryWarmCache, getWarmProgressUrl, getWarmStatus } from '../../../api/cache';

interface CacheWarmingCardProps {
  onWarm: (queries: string[]) => Promise<void>;
  onWarmingComplete?: () => void;
}

type TabType = 'manual' | 'file';

export function CacheWarmingCard({
  onWarm,
  onWarmingComplete,
}: CacheWarmingCardProps) {
  const [activeTab, setActiveTab] = useState<TabType>('manual');
  const [queryText, setQueryText] = useState('');
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);
  const [showFailedDetails, setShowFailedDetails] = useState(false);
  const eventSourceRef = useRef<EventSource | null>(null);

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
    completeWarming,
    setError,
    reset,
    startFileJob,
    updateProgress,
    addResult,
    completeFileJob,
    resetFileJob,
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

  // SSE connection
  const connectToSSE = useCallback((jobId: string) => {
    const url = getWarmProgressUrl(jobId);
    const eventSource = new EventSource(url);
    eventSourceRef.current = eventSource;

    eventSource.addEventListener('progress', (e) => {
      const data = JSON.parse(e.data);
      updateProgress(data.processed, data.estimated_remaining_seconds);
    });

    eventSource.addEventListener('result', (e) => {
      const data = JSON.parse(e.data);
      addResult(data);
    });

    eventSource.addEventListener('complete', () => {
      completeFileJob();
      eventSource.close();
      eventSourceRef.current = null;
    });

    eventSource.onerror = () => {
      setError('Connection to server lost. Please try again.');
      eventSource.close();
      eventSourceRef.current = null;
    };
  }, [updateProgress, addResult, completeFileJob, setError]);

  // Handle manual warming
  const handleManualWarm = async () => {
    if (queries.length === 0) return;

    startWarming(queries.length);

    try {
      await onWarm(queries);
      completeWarming();
      setQueryText('');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Warming failed');
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
      startFileJob(response.job_id, response.queued);
      connectToSSE(response.job_id);
    } catch (err) {
      setFileError(err instanceof Error ? err.message : 'Failed to start warming');
    }
  };

  // Handle retry failed
  const handleRetryFailed = async () => {
    if (failedQueries.length === 0) return;

    const queriesToRetry = [...failedQueries];
    resetFileJob();
    setFileError(null);

    try {
      const response = await retryWarmCache(queriesToRetry);
      startFileJob(response.job_id, response.queued);
      connectToSSE(response.job_id);
    } catch (err) {
      setFileError(err instanceof Error ? err.message : 'Retry failed');
    }
  };

  // Handle reset
  const handleReset = () => {
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

      {/* Status Messages */}
      {status === 'completed' && !isWarming && (
        <div className="flex items-center gap-2 p-3 mb-4 rounded-lg bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-400">
          <CheckCircle size={18} />
          <span className="text-sm flex-1">
            Successfully warmed {total > 0 ? `${total - failedQueries.length}/${total}` : queriesCount} queries
            {lastWarmingTime && ` at ${formatLastWarmingTime(lastWarmingTime)}`}
            {failedQueries.length > 0 && ` (${failedQueries.length} failed)`}
          </span>
          <button
            onClick={handleReset}
            className="text-xs text-green-600 dark:text-green-400 hover:underline"
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
            Upload a .txt or .csv file with one query per line. Numbered queries (e.g., "1. Question") are automatically cleaned.
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
    </Card>
  );
}
