import { useState, useEffect, useCallback, useRef } from 'react';
import { Plus, RefreshCw } from 'lucide-react';
import { Button, Alert, Select, Modal, Input, Pagination } from '../../ui';
import { MetricScoreCard } from './MetricScoreCard';
import { EvaluationRunCard } from './EvaluationRunCard';
import { EvaluationSampleTable } from './EvaluationSampleTable';
import { DatasetManager } from './DatasetManager';
import { LiveQualityChart } from './LiveQualityChart';
import { ConfirmModal } from './ConfirmModal';
import {
  getEvaluationSummary,
  getRuns,
  getDatasets,
  createRun,
  cancelRun,
  deleteRun,
} from '../../../api/evaluations';
import type {
  EvaluationSummary,
  EvaluationRun,
  EvaluationDataset,
  RunCreate,
} from '../../../types';

type SectionType = 'overview' | 'runs' | 'datasets' | 'live';

const SECTION_TABS: { id: SectionType; label: string }[] = [
  { id: 'overview', label: 'Overview' },
  { id: 'runs', label: 'Runs' },
  { id: 'datasets', label: 'Datasets' },
  { id: 'live', label: 'Live Monitoring' },
];

const STATUS_FILTER_OPTIONS = [
  { value: '', label: 'All Status' },
  { value: 'pending', label: 'Pending' },
  { value: 'running', label: 'Running' },
  { value: 'completed', label: 'Completed' },
  { value: 'failed', label: 'Failed' },
  { value: 'cancelled', label: 'Cancelled' },
];

const PAGE_SIZE = 10;
const AUTO_REFRESH_MS = 10_000;

export function EvaluationManager() {
  const [activeSection, setActiveSection] = useState<SectionType>('overview');
  const [summary, setSummary] = useState<EvaluationSummary | null>(null);
  const [runs, setRuns] = useState<EvaluationRun[]>([]);
  const [runsTotal, setRunsTotal] = useState(0);
  const [runsPage, setRunsPage] = useState(1);
  const [statusFilter, setStatusFilter] = useState('');
  const [loadingSummary, setLoadingSummary] = useState(true);
  const [loadingRuns, setLoadingRuns] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [actionLoading, setActionLoading] = useState(false);

  // Sample drill-down
  const [selectedRun, setSelectedRun] = useState<EvaluationRun | null>(null);
  const [showSamples, setShowSamples] = useState(false);

  // Create run modal
  const [showCreateRun, setShowCreateRun] = useState(false);
  const [createRunName, setCreateRunName] = useState('');
  const [createRunDesc, setCreateRunDesc] = useState('');
  const [createRunDatasetId, setCreateRunDatasetId] = useState('');
  const [datasetOptions, setDatasetOptions] = useState<EvaluationDataset[]>([]);

  // Delete run
  const [deletingRun, setDeletingRun] = useState<EvaluationRun | null>(null);

  const autoRefreshRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchSummary = useCallback(async () => {
    try {
      const data = await getEvaluationSummary();
      setSummary(data);
    } catch (err) {
      console.error('Failed to fetch summary:', err);
    } finally {
      setLoadingSummary(false);
    }
  }, []);

  const fetchRuns = useCallback(async () => {
    setLoadingRuns(true);
    try {
      const offset = (runsPage - 1) * PAGE_SIZE;
      const data = await getRuns(statusFilter || undefined, PAGE_SIZE, offset);
      setRuns(data.runs);
      setRunsTotal(data.total);
    } catch (err) {
      console.error('Failed to fetch runs:', err);
    } finally {
      setLoadingRuns(false);
    }
  }, [runsPage, statusFilter]);

  useEffect(() => {
    fetchSummary();
  }, [fetchSummary]);

  useEffect(() => {
    fetchRuns();
  }, [fetchRuns]);

  // Auto-refresh when active runs exist
  useEffect(() => {
    const hasActive = runs.some((r) => r.status === 'pending' || r.status === 'running');
    if (hasActive) {
      autoRefreshRef.current = setInterval(() => {
        fetchRuns();
        fetchSummary();
      }, AUTO_REFRESH_MS);
    }
    return () => {
      if (autoRefreshRef.current) clearInterval(autoRefreshRef.current);
    };
  }, [runs, fetchRuns, fetchSummary]);

  useEffect(() => {
    if (success) {
      const timer = setTimeout(() => setSuccess(null), 3000);
      return () => clearTimeout(timer);
    }
  }, [success]);

  const handleViewSamples = (run: EvaluationRun) => {
    setSelectedRun(run);
    setShowSamples(true);
  };

  const handleCancelRun = async (run: EvaluationRun) => {
    setActionLoading(true);
    try {
      await cancelRun(run.id);
      setSuccess(`Cancellation requested for "${run.name}"`);
      await fetchRuns();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to cancel run');
    } finally {
      setActionLoading(false);
    }
  };

  const handleConfirmDelete = async () => {
    if (!deletingRun) return;
    setActionLoading(true);
    try {
      await deleteRun(deletingRun.id);
      setSuccess(`Run "${deletingRun.name}" deleted`);
      setDeletingRun(null);
      await fetchRuns();
      await fetchSummary();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete run');
    } finally {
      setActionLoading(false);
    }
  };

  const openCreateRunModal = async () => {
    try {
      const data = await getDatasets(100, 0);
      setDatasetOptions(data.datasets);
    } catch {
      // ignore
    }
    setCreateRunName('');
    setCreateRunDesc('');
    setCreateRunDatasetId('');
    setShowCreateRun(true);
  };

  const handleCreateRun = async () => {
    if (!createRunName.trim() || !createRunDatasetId) return;
    setActionLoading(true);
    try {
      const data: RunCreate = {
        dataset_id: createRunDatasetId,
        name: createRunName.trim(),
        description: createRunDesc.trim() || null,
      };
      await createRun(data);
      setSuccess('Evaluation run started');
      setShowCreateRun(false);
      await fetchRuns();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create run');
    } finally {
      setActionLoading(false);
    }
  };

  const runsTotalPages = Math.ceil(runsTotal / PAGE_SIZE);

  return (
    <div className="space-y-6">
      {/* Sub-tab navigation */}
      <div className="border-b border-gray-200 dark:border-gray-700">
        <nav className="flex gap-4" aria-label="Evaluation sections">
          {SECTION_TABS.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveSection(tab.id)}
              className={`
                py-2 px-1 text-sm font-medium border-b-2 -mb-px transition-colors
                ${
                  activeSection === tab.id
                    ? 'border-primary text-primary'
                    : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
                }
              `}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </div>

      {error && (
        <Alert variant="danger" onClose={() => setError(null)}>
          {error}
        </Alert>
      )}
      {success && (
        <Alert variant="success" onClose={() => setSuccess(null)}>
          {success}
        </Alert>
      )}

      {/* Overview Section */}
      {activeSection === 'overview' && (
        <div className="space-y-6">
          {loadingSummary ? (
            <div className="py-8 text-center text-gray-500">Loading summary...</div>
          ) : summary ? (
            <>
              {/* Summary metric cards */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <MetricScoreCard label="Faithfulness" value={summary.avg_scores.faithfulness} />
                <MetricScoreCard label="Answer Relevancy" value={summary.avg_scores.answer_relevancy} />
                <MetricScoreCard label="Context Precision" value={summary.avg_scores.llm_context_precision} />
                <MetricScoreCard label="Context Recall" value={summary.avg_scores.llm_context_recall} />
              </div>

              {/* Counts */}
              <div className="grid grid-cols-2 gap-4">
                <div className="rounded-lg p-4 bg-gray-50 dark:bg-gray-800">
                  <span className="text-xs font-medium text-gray-500 dark:text-gray-400">
                    Total Runs
                  </span>
                  <div className="font-bold text-2xl text-gray-900 dark:text-white mt-1">
                    {summary.total_runs}
                  </div>
                </div>
                <div className="rounded-lg p-4 bg-gray-50 dark:bg-gray-800">
                  <span className="text-xs font-medium text-gray-500 dark:text-gray-400">
                    Total Datasets
                  </span>
                  <div className="font-bold text-2xl text-gray-900 dark:text-white mt-1">
                    {summary.total_datasets}
                  </div>
                </div>
              </div>

              {/* Latest run info */}
              {summary.latest_run && (
                <div>
                  <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Latest Run
                  </h3>
                  <EvaluationRunCard
                    run={summary.latest_run}
                    onViewSamples={handleViewSamples}
                    onCancel={handleCancelRun}
                    onDelete={(r) => setDeletingRun(r)}
                  />
                </div>
              )}
            </>
          ) : (
            <div className="py-8 text-center text-gray-500">
              No evaluation data yet. Create a dataset and start a run.
            </div>
          )}
        </div>
      )}

      {/* Runs Section */}
      {activeSection === 'runs' && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Select
                options={STATUS_FILTER_OPTIONS}
                value={statusFilter}
                onChange={(e) => {
                  setStatusFilter(e.target.value);
                  setRunsPage(1);
                }}
              />
              <span className="text-sm text-gray-500">{runsTotal} runs</span>
            </div>
            <div className="flex items-center gap-2">
              <Button variant="ghost" icon={RefreshCw} onClick={fetchRuns} disabled={loadingRuns} title="Refresh" />
              <Button icon={Plus} onClick={openCreateRunModal}>
                New Run
              </Button>
            </div>
          </div>

          {loadingRuns ? (
            <div className="py-8 text-center text-gray-500">Loading runs...</div>
          ) : runs.length === 0 ? (
            <div className="py-8 text-center text-gray-500">
              No evaluation runs found. Start one to begin evaluating.
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {runs.map((run) => (
                <EvaluationRunCard
                  key={run.id}
                  run={run}
                  onViewSamples={handleViewSamples}
                  onCancel={handleCancelRun}
                  onDelete={(r) => setDeletingRun(r)}
                />
              ))}
            </div>
          )}

          {runsTotalPages > 1 && (
            <Pagination
              currentPage={runsPage}
              totalPages={runsTotalPages}
              totalItems={runsTotal}
              itemsPerPage={PAGE_SIZE}
              onPageChange={setRunsPage}
            />
          )}
        </div>
      )}

      {/* Datasets Section */}
      {activeSection === 'datasets' && <DatasetManager />}

      {/* Live Monitoring Section */}
      {activeSection === 'live' && <LiveQualityChart />}

      {/* Sample drill-down modal */}
      <EvaluationSampleTable
        run={selectedRun}
        isOpen={showSamples}
        onClose={() => {
          setShowSamples(false);
          setSelectedRun(null);
        }}
      />

      {/* Create Run Modal */}
      <Modal isOpen={showCreateRun} onClose={() => setShowCreateRun(false)} title="Start Evaluation Run" size="md">
        <div className="space-y-4">
          <Input
            label="Run Name"
            value={createRunName}
            onChange={(e) => setCreateRunName(e.target.value)}
            placeholder="e.g., Baseline run v1"
          />
          <Input
            label="Description (optional)"
            value={createRunDesc}
            onChange={(e) => setCreateRunDesc(e.target.value)}
            placeholder="Description..."
          />
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Dataset
            </label>
            <select
              value={createRunDatasetId}
              onChange={(e) => setCreateRunDatasetId(e.target.value)}
              className="w-full rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-3 py-2 text-sm text-gray-900 dark:text-white"
            >
              <option value="">Select a dataset...</option>
              {datasetOptions.map((ds) => (
                <option key={ds.id} value={ds.id}>
                  {ds.name} ({ds.sample_count} samples)
                </option>
              ))}
            </select>
          </div>

          <div className="flex justify-end gap-3 pt-4 border-t border-gray-200 dark:border-gray-700">
            <Button variant="secondary" onClick={() => setShowCreateRun(false)}>
              Cancel
            </Button>
            <Button
              onClick={handleCreateRun}
              disabled={actionLoading || !createRunName.trim() || !createRunDatasetId}
            >
              {actionLoading ? 'Starting...' : 'Start Run'}
            </Button>
          </div>
        </div>
      </Modal>

      {/* Delete Run Confirmation */}
      <ConfirmModal
        isOpen={Boolean(deletingRun)}
        onClose={() => setDeletingRun(null)}
        onConfirm={handleConfirmDelete}
        title="Delete Evaluation Run"
        message={`Are you sure you want to delete "${deletingRun?.name}"? All sample results will be permanently removed.`}
        confirmLabel="Delete"
        isLoading={actionLoading}
        variant="danger"
      />
    </div>
  );
}
