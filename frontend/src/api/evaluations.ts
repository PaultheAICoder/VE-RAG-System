import { apiClient } from './client';
import type {
  DatasetListResponse,
  EvaluationDataset,
  DatasetCreate,
  RAGBenchImportRequest,
  SyntheticGenerateRequest,
  EvaluationSampleListResponse,
  RunListResponse,
  EvaluationRun,
  RunCreate,
  CancelRunResponse,
  EvaluationSummary,
  LiveStats,
  LiveScoreListResponse,
} from '../types';

const BASE = '/api/evaluations';

// ========== Datasets ==========

export async function getDatasets(limit = 20, offset = 0): Promise<DatasetListResponse> {
  return apiClient.get<DatasetListResponse>(`${BASE}/datasets?limit=${limit}&offset=${offset}`);
}

export async function getDataset(id: string): Promise<EvaluationDataset> {
  return apiClient.get<EvaluationDataset>(`${BASE}/datasets/${id}`);
}

export async function createDataset(data: DatasetCreate): Promise<EvaluationDataset> {
  return apiClient.post<EvaluationDataset>(`${BASE}/datasets`, data);
}

export async function deleteDataset(id: string): Promise<void> {
  await apiClient.delete(`${BASE}/datasets/${id}`);
}

export async function getDatasetSamples(
  id: string,
  limit = 50,
  offset = 0,
): Promise<EvaluationSampleListResponse> {
  return apiClient.get<EvaluationSampleListResponse>(
    `${BASE}/datasets/${id}/samples?limit=${limit}&offset=${offset}`,
  );
}

export async function importRAGBench(data: RAGBenchImportRequest): Promise<EvaluationDataset> {
  return apiClient.post<EvaluationDataset>(`${BASE}/datasets/import-ragbench`, data);
}

export async function generateSynthetic(
  data: SyntheticGenerateRequest,
): Promise<EvaluationDataset> {
  return apiClient.post<EvaluationDataset>(`${BASE}/datasets/generate-synthetic`, data);
}

// ========== Runs ==========

export async function getRuns(
  status?: string,
  limit = 20,
  offset = 0,
): Promise<RunListResponse> {
  const params = new URLSearchParams();
  params.set('limit', limit.toString());
  params.set('offset', offset.toString());
  if (status) params.set('status', status);
  return apiClient.get<RunListResponse>(`${BASE}/runs?${params.toString()}`);
}

export async function createRun(data: RunCreate): Promise<EvaluationRun> {
  return apiClient.post<EvaluationRun>(`${BASE}/runs`, data);
}

export async function getRun(id: string): Promise<EvaluationRun> {
  return apiClient.get<EvaluationRun>(`${BASE}/runs/${id}`);
}

export async function deleteRun(id: string): Promise<void> {
  await apiClient.delete(`${BASE}/runs/${id}`);
}

export async function cancelRun(id: string): Promise<CancelRunResponse> {
  return apiClient.post<CancelRunResponse>(`${BASE}/runs/${id}/cancel`);
}

export async function getRunSamples(
  runId: string,
  status?: string,
  limit = 20,
  offset = 0,
): Promise<EvaluationSampleListResponse> {
  const params = new URLSearchParams();
  params.set('limit', limit.toString());
  params.set('offset', offset.toString());
  if (status) params.set('status', status);
  return apiClient.get<EvaluationSampleListResponse>(
    `${BASE}/runs/${runId}/samples?${params.toString()}`,
  );
}

// ========== Summary ==========

export async function getEvaluationSummary(): Promise<EvaluationSummary> {
  return apiClient.get<EvaluationSummary>(`${BASE}/summary`);
}

// ========== Live Monitoring ==========

export async function getLiveStats(): Promise<LiveStats> {
  return apiClient.get<LiveStats>(`${BASE}/live/stats`);
}

export async function getLiveScores(limit = 20, offset = 0): Promise<LiveScoreListResponse> {
  return apiClient.get<LiveScoreListResponse>(
    `${BASE}/live/scores?limit=${limit}&offset=${offset}`,
  );
}
