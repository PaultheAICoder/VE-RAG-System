import { useState, useEffect, useCallback } from 'react';
import { Plus, RefreshCw, Trash2, Upload, Sparkles } from 'lucide-react';
import { Button, Alert, Card, Badge, Modal, Input, Pagination } from '../../ui';
import { ConfirmModal } from './ConfirmModal';
import {
  getDatasets,
  createDataset,
  deleteDataset,
  importRAGBench,
  generateSynthetic,
} from '../../../api/evaluations';
import type {
  EvaluationDataset,
  DatasetCreate,
  DatasetSampleCreate,
  RAGBenchImportRequest,
  SyntheticGenerateRequest,
} from '../../../types';

const SOURCE_BADGE: Record<string, 'default' | 'primary' | 'success' | 'warning'> = {
  manual: 'default',
  ragbench: 'primary',
  synthetic: 'success',
  live_sample: 'warning',
};

const PAGE_SIZE = 12;

export function DatasetManager() {
  const [datasets, setDatasets] = useState<EvaluationDataset[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [actionLoading, setActionLoading] = useState(false);

  const [showCreate, setShowCreate] = useState(false);
  const [showImport, setShowImport] = useState(false);
  const [showSynthetic, setShowSynthetic] = useState(false);
  const [deletingDataset, setDeletingDataset] = useState<EvaluationDataset | null>(null);

  // Create manual form state
  const [createName, setCreateName] = useState('');
  const [createDesc, setCreateDesc] = useState('');
  const [createSamples, setCreateSamples] = useState<DatasetSampleCreate[]>([
    { question: '', ground_truth: '' },
  ]);

  // Import RAGBench form state
  const [importSubset, setImportSubset] = useState('');
  const [importMaxSamples, setImportMaxSamples] = useState('50');
  const [importName, setImportName] = useState('');

  // Synthetic form state
  const [syntheticName, setSyntheticName] = useState('');
  const [syntheticDocIds, setSyntheticDocIds] = useState('');
  const [syntheticNumSamples, setSyntheticNumSamples] = useState('20');

  const fetchDatasets = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const offset = (page - 1) * PAGE_SIZE;
      const data = await getDatasets(PAGE_SIZE, offset);
      setDatasets(data.datasets);
      setTotal(data.total);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load datasets');
    } finally {
      setLoading(false);
    }
  }, [page]);

  useEffect(() => {
    fetchDatasets();
  }, [fetchDatasets]);

  useEffect(() => {
    if (success) {
      const timer = setTimeout(() => setSuccess(null), 3000);
      return () => clearTimeout(timer);
    }
  }, [success]);

  const resetCreateForm = () => {
    setCreateName('');
    setCreateDesc('');
    setCreateSamples([{ question: '', ground_truth: '' }]);
  };

  const handleCreateManual = async () => {
    if (!createName.trim() || createSamples.every((s) => !s.question.trim())) return;
    setActionLoading(true);
    try {
      const data: DatasetCreate = {
        name: createName.trim(),
        description: createDesc.trim() || null,
        samples: createSamples.filter((s) => s.question.trim()),
      };
      await createDataset(data);
      setSuccess('Dataset created successfully');
      setShowCreate(false);
      resetCreateForm();
      await fetchDatasets();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create dataset');
    } finally {
      setActionLoading(false);
    }
  };

  const handleImportRAGBench = async () => {
    if (!importSubset.trim() || !importName.trim()) return;
    setActionLoading(true);
    try {
      const data: RAGBenchImportRequest = {
        subset: importSubset.trim(),
        max_samples: parseInt(importMaxSamples) || 50,
        name: importName.trim(),
      };
      await importRAGBench(data);
      setSuccess('RAGBench dataset imported successfully');
      setShowImport(false);
      setImportSubset('');
      setImportMaxSamples('50');
      setImportName('');
      await fetchDatasets();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to import RAGBench dataset');
    } finally {
      setActionLoading(false);
    }
  };

  const handleGenerateSynthetic = async () => {
    if (!syntheticName.trim() || !syntheticDocIds.trim()) return;
    setActionLoading(true);
    try {
      const data: SyntheticGenerateRequest = {
        name: syntheticName.trim(),
        document_ids: syntheticDocIds
          .split(',')
          .map((s) => s.trim())
          .filter(Boolean),
        num_samples: parseInt(syntheticNumSamples) || 20,
      };
      await generateSynthetic(data);
      setSuccess('Synthetic dataset generated successfully');
      setShowSynthetic(false);
      setSyntheticName('');
      setSyntheticDocIds('');
      setSyntheticNumSamples('20');
      await fetchDatasets();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate synthetic dataset');
    } finally {
      setActionLoading(false);
    }
  };

  const handleConfirmDelete = async () => {
    if (!deletingDataset) return;
    setActionLoading(true);
    try {
      await deleteDataset(deletingDataset.id);
      setSuccess('Dataset deleted successfully');
      setDeletingDataset(null);
      await fetchDatasets();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete dataset');
    } finally {
      setActionLoading(false);
    }
  };

  const addSampleRow = () => {
    setCreateSamples([...createSamples, { question: '', ground_truth: '' }]);
  };

  const updateSample = (index: number, field: 'question' | 'ground_truth', value: string) => {
    const updated = [...createSamples];
    updated[index] = { ...updated[index], [field]: value };
    setCreateSamples(updated);
  };

  const removeSampleRow = (index: number) => {
    if (createSamples.length <= 1) return;
    setCreateSamples(createSamples.filter((_, i) => i !== index));
  };

  const totalPages = Math.ceil(total / PAGE_SIZE);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-heading font-bold text-gray-900 dark:text-white">Datasets</h2>
          <p className="text-gray-500 dark:text-gray-400 mt-1">
            Manage evaluation datasets for benchmark runs
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="ghost" icon={RefreshCw} onClick={fetchDatasets} disabled={loading} title="Refresh" />
          <Button variant="outline" icon={Upload} onClick={() => setShowImport(true)}>
            Import RAGBench
          </Button>
          <Button variant="outline" icon={Sparkles} onClick={() => setShowSynthetic(true)}>
            Generate Synthetic
          </Button>
          <Button icon={Plus} onClick={() => setShowCreate(true)}>
            Create Manual
          </Button>
        </div>
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

      {loading ? (
        <div className="py-12 text-center text-gray-500">Loading datasets...</div>
      ) : datasets.length === 0 ? (
        <div className="py-12 text-center text-gray-500">
          No datasets yet. Create or import one to get started.
        </div>
      ) : (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {datasets.map((ds) => (
              <Card key={ds.id} className="hover:shadow-md transition-shadow">
                <div className="flex items-start justify-between mb-2">
                  <h3 className="font-medium text-gray-900 dark:text-white truncate pr-2">{ds.name}</h3>
                  <Badge variant={SOURCE_BADGE[ds.source_type] || 'default'}>{ds.source_type}</Badge>
                </div>
                {ds.description && (
                  <p className="text-sm text-gray-500 dark:text-gray-400 mb-2 line-clamp-2">{ds.description}</p>
                )}
                <div className="text-sm text-gray-500 dark:text-gray-400 mb-3">
                  {ds.sample_count} samples
                </div>
                <div className="flex items-center justify-between pt-2 border-t border-gray-100 dark:border-gray-700">
                  <span className="text-xs text-gray-400">
                    {new Date(ds.created_at).toLocaleDateString()}
                  </span>
                  <Button
                    size="sm"
                    variant="danger"
                    icon={Trash2}
                    onClick={() => setDeletingDataset(ds)}
                  >
                    Delete
                  </Button>
                </div>
              </Card>
            ))}
          </div>

          {totalPages > 1 && (
            <Pagination
              currentPage={page}
              totalPages={totalPages}
              totalItems={total}
              itemsPerPage={PAGE_SIZE}
              onPageChange={setPage}
            />
          )}
        </>
      )}

      {/* Create Manual Dataset Modal */}
      <Modal isOpen={showCreate} onClose={() => setShowCreate(false)} title="Create Manual Dataset" size="lg">
        <div className="space-y-4">
          <Input label="Name" value={createName} onChange={(e) => setCreateName(e.target.value)} placeholder="My evaluation dataset" />
          <Input label="Description (optional)" value={createDesc} onChange={(e) => setCreateDesc(e.target.value)} placeholder="Description..." />

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Samples
            </label>
            <div className="space-y-2 max-h-60 overflow-y-auto">
              {createSamples.map((sample, idx) => (
                <div key={idx} className="flex gap-2 items-start">
                  <Input
                    placeholder="Question"
                    value={sample.question}
                    onChange={(e) => updateSample(idx, 'question', e.target.value)}
                    className="flex-1"
                  />
                  <Input
                    placeholder="Ground truth (optional)"
                    value={sample.ground_truth || ''}
                    onChange={(e) => updateSample(idx, 'ground_truth', e.target.value)}
                    className="flex-1"
                  />
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => removeSampleRow(idx)}
                    disabled={createSamples.length <= 1}
                  >
                    X
                  </Button>
                </div>
              ))}
            </div>
            <Button size="sm" variant="outline" onClick={addSampleRow} className="mt-2">
              + Add Sample
            </Button>
          </div>

          <div className="flex justify-end gap-3 pt-4 border-t border-gray-200 dark:border-gray-700">
            <Button variant="secondary" onClick={() => setShowCreate(false)}>
              Cancel
            </Button>
            <Button onClick={handleCreateManual} disabled={actionLoading || !createName.trim()}>
              {actionLoading ? 'Creating...' : 'Create Dataset'}
            </Button>
          </div>
        </div>
      </Modal>

      {/* Import RAGBench Modal */}
      <Modal isOpen={showImport} onClose={() => setShowImport(false)} title="Import RAGBench Dataset" size="md">
        <div className="space-y-4">
          <Input label="Subset Name" value={importSubset} onChange={(e) => setImportSubset(e.target.value)} placeholder="e.g., hotpotqa" />
          <Input label="Dataset Name" value={importName} onChange={(e) => setImportName(e.target.value)} placeholder="Name for this dataset" />
          <Input label="Max Samples" type="number" value={importMaxSamples} onChange={(e) => setImportMaxSamples(e.target.value)} placeholder="50" />

          <div className="flex justify-end gap-3 pt-4 border-t border-gray-200 dark:border-gray-700">
            <Button variant="secondary" onClick={() => setShowImport(false)}>
              Cancel
            </Button>
            <Button onClick={handleImportRAGBench} disabled={actionLoading || !importSubset.trim() || !importName.trim()}>
              {actionLoading ? 'Importing...' : 'Import'}
            </Button>
          </div>
        </div>
      </Modal>

      {/* Generate Synthetic Modal */}
      <Modal isOpen={showSynthetic} onClose={() => setShowSynthetic(false)} title="Generate Synthetic Dataset" size="md">
        <div className="space-y-4">
          <Input label="Dataset Name" value={syntheticName} onChange={(e) => setSyntheticName(e.target.value)} placeholder="Name for this dataset" />
          <Input label="Document IDs (comma-separated)" value={syntheticDocIds} onChange={(e) => setSyntheticDocIds(e.target.value)} placeholder="doc-id-1, doc-id-2" />
          <Input label="Number of Samples" type="number" value={syntheticNumSamples} onChange={(e) => setSyntheticNumSamples(e.target.value)} placeholder="20" />

          <div className="flex justify-end gap-3 pt-4 border-t border-gray-200 dark:border-gray-700">
            <Button variant="secondary" onClick={() => setShowSynthetic(false)}>
              Cancel
            </Button>
            <Button onClick={handleGenerateSynthetic} disabled={actionLoading || !syntheticName.trim() || !syntheticDocIds.trim()}>
              {actionLoading ? 'Generating...' : 'Generate'}
            </Button>
          </div>
        </div>
      </Modal>

      {/* Delete Confirmation */}
      <ConfirmModal
        isOpen={Boolean(deletingDataset)}
        onClose={() => setDeletingDataset(null)}
        onConfirm={handleConfirmDelete}
        title="Delete Dataset"
        message={`Are you sure you want to delete "${deletingDataset?.name}"? This cannot be undone. Datasets with existing runs cannot be deleted.`}
        confirmLabel="Delete"
        isLoading={actionLoading}
        variant="danger"
      />
    </div>
  );
}
