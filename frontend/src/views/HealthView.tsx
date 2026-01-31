import { useState, useEffect, useCallback, useRef } from 'react';
import { RefreshCw, Server, Cpu, Database, HardDrive, Clock } from 'lucide-react';
import { Button, Alert } from '../components/ui';
import { HealthCard, StatsCard, PipelineVisualization } from '../components/features/admin';
import { getHealth } from '../api/health';
import { getArchitectureInfo, getKnowledgeBaseStats } from '../api/admin';
import type { HealthResponse, ArchitectureInfo, KnowledgeBaseStats } from '../types';

const AUTO_REFRESH_INTERVAL = 30000; // 30 seconds

function formatBytes(bytes: number | null | undefined): string {
  if (bytes == null) return 'Unknown';
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
}

function formatRelativeTime(lastUpdated: Date): string {
  const seconds = Math.floor((Date.now() - lastUpdated.getTime()) / 1000);
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ago`;
}

export function HealthView() {
  // Data state
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [architecture, setArchitecture] = useState<ArchitectureInfo | null>(null);
  const [kbStats, setKbStats] = useState<KnowledgeBaseStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date>(new Date());
  const [relativeTime, setRelativeTime] = useState('just now');

  const refreshIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const timeIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Fetch all health data
  const fetchData = useCallback(async () => {
    setError(null);
    try {
      const [healthData, archData, statsData] = await Promise.all([
        getHealth(),
        getArchitectureInfo(),
        getKnowledgeBaseStats(),
      ]);
      setHealth(healthData);
      setArchitecture(archData);
      setKbStats(statsData);
      setLastUpdated(new Date());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load health data');
    } finally {
      setLoading(false);
    }
  }, []);

  // Initial fetch
  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Auto-refresh
  useEffect(() => {
    refreshIntervalRef.current = setInterval(fetchData, AUTO_REFRESH_INTERVAL);
    return () => {
      if (refreshIntervalRef.current) {
        clearInterval(refreshIntervalRef.current);
      }
    };
  }, [fetchData]);

  // Update relative time display
  useEffect(() => {
    const updateTime = () => {
      setRelativeTime(formatRelativeTime(lastUpdated));
    };
    updateTime();
    timeIntervalRef.current = setInterval(updateTime, 1000);
    return () => {
      if (timeIntervalRef.current) {
        clearInterval(timeIntervalRef.current);
      }
    };
  }, [lastUpdated]);

  // Manual refresh
  const handleRefresh = () => {
    setLoading(true);
    fetchData();
  };

  // Determine component health
  const apiHealthy = health?.status === 'healthy';
  const ollamaHealthy = architecture?.infrastructure.ollama_status === 'healthy';
  const vectorDbHealthy = architecture?.infrastructure.vector_db_status === 'healthy';

  // Build pipeline stages
  const pipelineStages = [
    { name: 'Query', healthy: apiHealthy },
    { name: 'Embed', healthy: ollamaHealthy },
    { name: 'Search', healthy: vectorDbHealthy },
    { name: 'Rerank', healthy: true }, // Assume rerank works if search works
    { name: 'Context', healthy: true },
    { name: 'LLM', healthy: ollamaHealthy },
    { name: 'Response', healthy: apiHealthy },
  ];

  if (loading && !health) {
    return (
      <div className="p-6 flex items-center justify-center h-64">
        <div className="text-gray-500">Loading health data...</div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-heading font-bold text-gray-900 dark:text-white">
            System Health
          </h1>
          <p className="text-gray-500 dark:text-gray-400 mt-1">
            Monitor system status and performance
          </p>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 text-sm text-gray-500">
            <Clock size={16} />
            <span>Last checked: {relativeTime}</span>
          </div>
          <Button icon={RefreshCw} variant="secondary" onClick={handleRefresh} disabled={loading}>
            {loading ? 'Refreshing...' : 'Refresh'}
          </Button>
        </div>
      </div>

      {/* Error Alert */}
      {error && (
        <Alert variant="danger" onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <HealthCard
          title="API Server"
          status={apiHealthy ? 'healthy' : 'unhealthy'}
          icon={<Server size={24} />}
          details={[
            { label: 'Version', value: health?.version || 'Unknown' },
            { label: 'Profile', value: health?.profile || 'Unknown' },
            { label: 'Database', value: health?.database || 'Unknown' },
          ]}
        />
        <HealthCard
          title="Ollama LLM"
          status={ollamaHealthy ? 'healthy' : 'unhealthy'}
          icon={<Cpu size={24} />}
          details={[
            { label: 'Model', value: architecture?.chat_model.name || 'Unknown' },
            { label: 'URL', value: architecture?.infrastructure.ollama_url || 'Unknown' },
            { label: 'Status', value: architecture?.infrastructure.ollama_status || 'Unknown' },
          ]}
        />
        <HealthCard
          title="Vector DB"
          status={vectorDbHealthy ? 'healthy' : 'unhealthy'}
          icon={<Database size={24} />}
          details={[
            { label: 'Backend', value: architecture?.embeddings.vector_store || 'Unknown' },
            { label: 'Chunks', value: kbStats?.total_chunks.toLocaleString() || '0' },
            {
              label: 'Model',
              value: architecture?.embeddings.model || 'Unknown',
            },
          ]}
        />
      </div>

      {/* Pipeline Visualization */}
      <PipelineVisualization
        stages={pipelineStages}
        embeddingModel={
          architecture?.embeddings.model
            ? `${architecture.embeddings.model} (${architecture.embeddings.dimensions} dim)`
            : undefined
        }
        chatModel={architecture?.chat_model.name}
        chunker={`${architecture?.document_parsing.engine || 'Unknown'} (${architecture?.profile || 'Unknown'} profile)`}
      />

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <StatsCard
          title="Knowledge Base"
          icon={<HardDrive size={20} />}
          stats={[
            { label: 'Total Documents', value: kbStats?.unique_files || 0 },
            { label: 'Total Chunks', value: kbStats?.total_chunks.toLocaleString() || '0' },
            { label: 'Storage Used', value: formatBytes(kbStats?.storage_size_bytes) },
            { label: 'Collection', value: kbStats?.collection_name || 'Unknown' },
          ]}
        />
        <StatsCard
          title="System Configuration"
          icon={<Server size={20} />}
          stats={[
            { label: 'Profile', value: architecture?.profile || 'Unknown' },
            { label: 'Vector Backend', value: architecture?.embeddings.vector_store || 'Unknown' },
            { label: 'Chunker', value: architecture?.document_parsing.engine || 'Unknown' },
            { label: 'RAG Enabled', value: health?.rag_enabled ? 'Yes' : 'No' },
          ]}
        />
      </div>

      {/* OCR Status */}
      {architecture?.ocr_status && (
        <div className="bg-gray-50 dark:bg-gray-800/50 rounded-xl p-4">
          <h3 className="font-semibold text-gray-900 dark:text-white mb-3">OCR Status</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="flex items-center gap-3">
              <span
                className={`w-2.5 h-2.5 rounded-full ${
                  architecture.ocr_status.tesseract.available ? 'bg-green-500' : 'bg-gray-400'
                }`}
              />
              <div>
                <span className="font-medium text-gray-700 dark:text-gray-300">Tesseract</span>
                {architecture.ocr_status.tesseract.available ? (
                  <span className="text-sm text-gray-500 ml-2">
                    v{architecture.ocr_status.tesseract.version} -{' '}
                    {architecture.ocr_status.tesseract.languages?.length || 0} languages
                  </span>
                ) : (
                  <span className="text-sm text-gray-400 ml-2">Not installed</span>
                )}
              </div>
            </div>
            <div className="flex items-center gap-3">
              <span
                className={`w-2.5 h-2.5 rounded-full ${
                  architecture.ocr_status.easyocr.available ? 'bg-green-500' : 'bg-gray-400'
                }`}
              />
              <div>
                <span className="font-medium text-gray-700 dark:text-gray-300">EasyOCR</span>
                {architecture.ocr_status.easyocr.available ? (
                  <span className="text-sm text-gray-500 ml-2">
                    v{architecture.ocr_status.easyocr.version}
                  </span>
                ) : (
                  <span className="text-sm text-gray-400 ml-2">Not installed</span>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
