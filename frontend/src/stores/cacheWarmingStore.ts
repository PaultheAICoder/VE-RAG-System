import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { WarmingJob, WarmingQuery } from '../types';

type WarmingStatus = 'idle' | 'warming' | 'completed' | 'error';

interface QueryResult {
  query: string;
  status: 'success' | 'failed';
  cached: boolean;
  error: string | null;
}

interface CacheWarmingState {
  // Manual warming state
  status: WarmingStatus;
  queriesCount: number;
  lastWarmingTime: string | null;
  error: string | null;

  // File upload job state
  activeJobId: string | null;
  processed: number;
  total: number;
  results: QueryResult[];
  failedQueries: string[];
  estimatedTimeRemaining: number | null;

  // Actions - Manual warming
  startWarming: (queriesCount: number) => void;
  completeWarming: () => void;
  setError: (error: string) => void;
  reset: () => void;

  // Actions - File upload
  startFileJob: (jobId: string, total: number) => void;
  updateProgress: (data: {
    processed: number;
    failed: number;
    skipped: number;
    total: number;
    percent: number;
    batch_status: string;
  }) => void;
  addResult: (result: QueryResult) => void;
  completeFileJob: () => void;
  resetFileJob: () => void;

  // Queue management state
  queueJobs: WarmingJob[];
  queueLoading: boolean;
  queueError: string | null;

  // Bulk selection state
  selectedJobIds: Set<string>;

  // SSE reconnection
  lastEventId: string | null;

  // Query expansion state
  expandedBatchQueries: Record<string, WarmingQuery[]>;
  expandedBatchLoading: Record<string, boolean>;

  // Queue management actions
  setQueueJobs: (jobs: WarmingJob[]) => void;
  setQueueLoading: (loading: boolean) => void;
  setQueueError: (error: string | null) => void;
  updateQueueJob: (job: WarmingJob) => void;
  removeQueueJob: (jobId: string) => void;

  // Bulk selection actions
  setSelectedJobIds: (ids: Set<string>) => void;
  toggleJobSelection: (jobId: string) => void;
  selectAllJobs: () => void;
  clearSelection: () => void;

  // SSE reconnection actions
  setLastEventId: (id: string | null) => void;

  // Query expansion actions
  setExpandedBatchQueries: (batchId: string, queries: WarmingQuery[]) => void;
  setExpandedBatchLoading: (batchId: string, loading: boolean) => void;
  clearExpandedBatch: (batchId: string) => void;
}

export const useCacheWarmingStore = create<CacheWarmingState>()(
  persist(
    (set) => ({
      // Initial state - Manual
      status: 'idle',
      queriesCount: 0,
      lastWarmingTime: null,
      error: null,

      // Initial state - File upload
      activeJobId: null,
      processed: 0,
      total: 0,
      results: [],
      failedQueries: [],
      estimatedTimeRemaining: null,

      // Initial state - Queue management
      queueJobs: [],
      queueLoading: false,
      queueError: null,

      // Initial state - Bulk selection
      selectedJobIds: new Set<string>(),

      // Initial state - SSE reconnection
      lastEventId: null,

      // Initial state - Query expansion
      expandedBatchQueries: {},
      expandedBatchLoading: {},

      // Manual warming actions
      startWarming: (queriesCount) =>
        set({
          status: 'warming',
          queriesCount,
          error: null,
        }),

      completeWarming: () =>
        set({
          status: 'completed',
          lastWarmingTime: new Date().toISOString(),
        }),

      setError: (error) =>
        set({
          status: 'error',
          error,
        }),

      reset: () =>
        set({
          status: 'idle',
          queriesCount: 0,
          error: null,
          activeJobId: null,
          processed: 0,
          total: 0,
          results: [],
          failedQueries: [],
          estimatedTimeRemaining: null,
        }),

      // File upload actions
      startFileJob: (jobId, total) =>
        set({
          status: 'warming',
          activeJobId: jobId,
          processed: 0,
          total,
          results: [],
          failedQueries: [],
          estimatedTimeRemaining: null,
          error: null,
        }),

      updateProgress: (data) =>
        set({
          processed: data.processed + data.failed + data.skipped,
          total: data.total,
          estimatedTimeRemaining: null,
        }),

      addResult: (result) =>
        set((state) => ({
          results: [...state.results, result],
          failedQueries:
            result.status === 'failed'
              ? [...state.failedQueries, result.query]
              : state.failedQueries,
        })),

      completeFileJob: () =>
        set({
          status: 'completed',
          lastWarmingTime: new Date().toISOString(),
          activeJobId: null,
          estimatedTimeRemaining: null,
        }),

      resetFileJob: () =>
        set({
          status: 'idle',
          activeJobId: null,
          processed: 0,
          total: 0,
          results: [],
          failedQueries: [],
          estimatedTimeRemaining: null,
          error: null,
        }),

      // Queue management actions
      setQueueJobs: (jobs) => set({ queueJobs: jobs }),
      setQueueLoading: (loading) => set({ queueLoading: loading }),
      setQueueError: (error) => set({ queueError: error }),
      updateQueueJob: (job) =>
        set((state) => ({
          queueJobs: state.queueJobs.map((j) => (j.id === job.id ? job : j)),
        })),
      removeQueueJob: (jobId) =>
        set((state) => ({
          queueJobs: state.queueJobs.filter((j) => j.id !== jobId),
          selectedJobIds: new Set([...state.selectedJobIds].filter((id) => id !== jobId)),
        })),

      // Bulk selection actions
      setSelectedJobIds: (ids) => set({ selectedJobIds: ids }),
      toggleJobSelection: (jobId) =>
        set((state) => {
          const newSet = new Set(state.selectedJobIds);
          if (newSet.has(jobId)) {
            newSet.delete(jobId);
          } else {
            newSet.add(jobId);
          }
          return { selectedJobIds: newSet };
        }),
      selectAllJobs: () =>
        set((state) => ({
          selectedJobIds: new Set(state.queueJobs.map((j) => j.id)),
        })),
      clearSelection: () => set({ selectedJobIds: new Set<string>() }),

      // Query expansion actions
      setExpandedBatchQueries: (batchId, queries) =>
        set((state) => ({
          expandedBatchQueries: { ...state.expandedBatchQueries, [batchId]: queries },
        })),
      setExpandedBatchLoading: (batchId, loading) =>
        set((state) => ({
          expandedBatchLoading: { ...state.expandedBatchLoading, [batchId]: loading },
        })),
      clearExpandedBatch: (batchId) =>
        set((state) => {
          const nextQueries = { ...state.expandedBatchQueries };
          delete nextQueries[batchId];
          const nextLoading = { ...state.expandedBatchLoading };
          delete nextLoading[batchId];
          return { expandedBatchQueries: nextQueries, expandedBatchLoading: nextLoading };
        }),

      // SSE reconnection actions
      setLastEventId: (id) => set({ lastEventId: id }),
    }),
    {
      name: 'cache-warming-storage',
      partialize: (state) => ({
        status: state.status,
        queriesCount: state.queriesCount,
        lastWarmingTime: state.lastWarmingTime,
        activeJobId: state.activeJobId,
        processed: state.processed,
        total: state.total,
        failedQueries: state.failedQueries,
        lastEventId: state.lastEventId,
      }),
    }
  )
);
