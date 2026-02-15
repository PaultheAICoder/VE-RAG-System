import { useState, useEffect, useCallback } from 'react';
import { Modal, Badge, Pagination, Select } from '../../ui';
import { getRunSamples } from '../../../api/evaluations';
import type { EvaluationRun, EvaluationSample } from '../../../types';

interface EvaluationSampleTableProps {
  run: EvaluationRun | null;
  isOpen: boolean;
  onClose: () => void;
}

const STATUS_FILTER_OPTIONS = [
  { value: '', label: 'All Status' },
  { value: 'pending', label: 'Pending' },
  { value: 'completed', label: 'Completed' },
  { value: 'failed', label: 'Failed' },
];

const STATUS_VARIANT: Record<string, 'default' | 'primary' | 'success' | 'warning' | 'danger'> = {
  pending: 'default',
  running: 'primary',
  completed: 'success',
  failed: 'danger',
};

const PAGE_SIZE = 20;

function formatScore(v: number | null): string {
  if (v === null) return '—';
  return (v * 100).toFixed(1) + '%';
}

function scoreColor(v: number | null): string {
  if (v === null) return 'text-gray-400';
  if (v >= 0.7) return 'text-green-600 dark:text-green-400';
  if (v >= 0.4) return 'text-amber-600 dark:text-amber-400';
  return 'text-red-600 dark:text-red-400';
}

export function EvaluationSampleTable({ run, isOpen, onClose }: EvaluationSampleTableProps) {
  const [samples, setSamples] = useState<EvaluationSample[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [statusFilter, setStatusFilter] = useState('');
  const [loading, setLoading] = useState(false);

  const fetchSamples = useCallback(async () => {
    if (!run) return;
    setLoading(true);
    try {
      const offset = (page - 1) * PAGE_SIZE;
      const data = await getRunSamples(run.id, statusFilter || undefined, PAGE_SIZE, offset);
      setSamples(data.samples);
      setTotal(data.total);
    } catch (err) {
      console.error('Failed to fetch samples:', err);
    } finally {
      setLoading(false);
    }
  }, [run, page, statusFilter]);

  useEffect(() => {
    if (isOpen && run) {
      setPage(1);
      fetchSamples();
    }
  }, [isOpen, run?.id, statusFilter]);

  useEffect(() => {
    if (isOpen && run) fetchSamples();
  }, [page]);

  const totalPages = Math.ceil(total / PAGE_SIZE);

  return (
    <Modal isOpen={isOpen} onClose={onClose} title={`Samples — ${run?.name || ''}`} size="xl">
      <div className="space-y-4">
        <div className="flex items-center gap-4">
          <Select
            label="Status"
            options={STATUS_FILTER_OPTIONS}
            value={statusFilter}
            onChange={(e) => {
              setStatusFilter(e.target.value);
              setPage(1);
            }}
          />
          <span className="text-sm text-gray-500">{total} samples</span>
        </div>

        {loading ? (
          <div className="text-center py-8 text-gray-500">Loading...</div>
        ) : samples.length === 0 ? (
          <div className="text-center py-8 text-gray-500">No samples found</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700 text-left">
                  <th className="py-2 px-2 font-medium text-gray-500">#</th>
                  <th className="py-2 px-2 font-medium text-gray-500">Question</th>
                  <th className="py-2 px-2 font-medium text-gray-500">Status</th>
                  <th className="py-2 px-2 font-medium text-gray-500 text-right">Faith.</th>
                  <th className="py-2 px-2 font-medium text-gray-500 text-right">Relev.</th>
                  <th className="py-2 px-2 font-medium text-gray-500 text-right">Prec.</th>
                  <th className="py-2 px-2 font-medium text-gray-500 text-right">Recall</th>
                  <th className="py-2 px-2 font-medium text-gray-500 text-right">Time</th>
                </tr>
              </thead>
              <tbody>
                {samples.map((s) => (
                  <tr
                    key={s.id}
                    className="border-b border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-800/50"
                  >
                    <td className="py-2 px-2 text-gray-400">{s.sort_order + 1}</td>
                    <td className="py-2 px-2 max-w-xs truncate" title={s.question}>
                      {s.question}
                    </td>
                    <td className="py-2 px-2">
                      <Badge variant={STATUS_VARIANT[s.status] || 'default'}>{s.status}</Badge>
                    </td>
                    <td className={`py-2 px-2 text-right font-mono ${scoreColor(s.faithfulness)}`}>
                      {formatScore(s.faithfulness)}
                    </td>
                    <td className={`py-2 px-2 text-right font-mono ${scoreColor(s.answer_relevancy)}`}>
                      {formatScore(s.answer_relevancy)}
                    </td>
                    <td className={`py-2 px-2 text-right font-mono ${scoreColor(s.llm_context_precision)}`}>
                      {formatScore(s.llm_context_precision)}
                    </td>
                    <td className={`py-2 px-2 text-right font-mono ${scoreColor(s.llm_context_recall)}`}>
                      {formatScore(s.llm_context_recall)}
                    </td>
                    <td className="py-2 px-2 text-right text-gray-500">
                      {s.generation_time_ms !== null ? `${Math.round(s.generation_time_ms)}ms` : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {totalPages > 1 && (
          <Pagination
            currentPage={page}
            totalPages={totalPages}
            totalItems={total}
            itemsPerPage={PAGE_SIZE}
            onPageChange={setPage}
          />
        )}
      </div>
    </Modal>
  );
}
