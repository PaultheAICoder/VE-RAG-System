import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { WarmingJob } from '../types';

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
  updateProgress: (processed: number, estimatedRemaining?: number) => void;
  addResult: (result: QueryResult) => void;
  completeFileJob: () => void;
  resetFileJob: () => void;

  // Queue management state
  queueJobs: WarmingJob[];
  queueLoading: boolean;
  queueError: string | null;

  // Queue management actions
  setQueueJobs: (jobs: WarmingJob[]) => void;
  setQueueLoading: (loading: boolean) => void;
  setQueueError: (error: string | null) => void;
  updateQueueJob: (job: WarmingJob) => void;
  removeQueueJob: (jobId: string) => void;
}

export const useCacheWarmingStore = create<CacheWarmingState>()(
  persist(
    (set, get) => ({
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

      updateProgress: (processed, estimatedRemaining) =>
        set({
          processed,
          estimatedTimeRemaining: estimatedRemaining ?? get().estimatedTimeRemaining,
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
        })),
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
      }),
    }
  )
);
