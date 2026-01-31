import { useState, useEffect, useCallback, useRef } from 'react';
import { RefreshCw, Server, Cpu, Database, HardDrive, Clock, Activity, Trash2 } from 'lucide-react';
import { Button, Alert } from '../components/ui';
import { HealthCard, StatsCard, PipelineVisualization, ClearKBModal } from '../components/features/admin';
import { getDetailedHealth } from '../api/health';
import { clearKnowledgeBase } from '../api/admin';
import type { DetailedHealthResponse } from '../types';

const AUTO_REFRESH_INTERVAL_NORMAL = 30000; // 30 seconds when idle
const AUTO_REFRESH_INTERVAL_ACTIVE = 3000; // 3 seconds when processing

function formatRelativeTime(lastUpdated: Date): string {
  const seconds = Math.floor((Date.now() - lastUpdated.getTime()) / 1000);
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ago`;
}

function formatUptime(seconds: number): string {
  const days = Math.floor(seconds / 86400);
  const hours = Math.floor((seconds % 86400) / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  if (days > 0) return `${days}d ${hours}h ${minutes}m`;
  if (hours > 0) return `${hours}h ${minutes}m`;
  return `${minutes}m`;
}

export function HealthView() {
  // Data state - single endpoint provides all data
  const [healthData, setHealthData] = useState<DetailedHealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date>(new Date());
  const [relativeTime, setRelativeTime] = useState('just now');

  // Clear KB modal state
  const [showClearModal, setShowClearModal] = useState(false);
  const [clearing, setClearing] = useState(false);

  const refreshIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const timeIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Fetch all health data from single detailed endpoint
  const fetchData = useCallback(async () => {
    setError(null);
    try {
      const data = await getDetailedHealth();
      setHealthData(data);
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

  // Determine if processing is active (pending or processing documents)
  const isProcessingActive =
    (healthData?.processing_queue.pending || 0) > 0 ||
    (healthData?.processing_queue.processing || 0) > 0;

  // Auto-refresh with dynamic interval
  useEffect(() => {
    const interval = isProcessingActive
      ? AUTO_REFRESH_INTERVAL_ACTIVE
      : AUTO_REFRESH_INTERVAL_NORMAL;

    refreshIntervalRef.current = setInterval(fetchData, interval);
    return () => {
      if (refreshIntervalRef.current) {
        clearInterval(refreshIntervalRef.current);
      }
    };
  }, [fetchData, isProcessingActive]);

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

  // Clear knowledge base
  const handleClearKB = async (deleteSourceFiles: boolean) => {
    setClearing(true);
    try {
      await clearKnowledgeBase(deleteSourceFiles);
      setShowClearModal(false);
      // Refresh data to show updated stats
      fetchData();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to clear knowledge base');
    } finally {
      setClearing(false);
    }
  };

  // Determine component health from detailed response
  const apiHealthy = healthData?.api_server.status === 'healthy';
  const ollamaHealthy = healthData?.ollama_llm.status === 'healthy';
  const vectorDbHealthy = healthData?.vector_db.status === 'healthy';

  // Build pipeline stages from rag_pipeline
  const pipelineStages = healthData?.rag_pipeline.stages.map((stage) => ({
    name: stage,
    healthy: healthData.rag_pipeline.all_stages_healthy,
  })) || [
    { name: 'Query', healthy: apiHealthy },
    { name: 'Embed', healthy: ollamaHealthy },
    { name: 'Search', healthy: vectorDbHealthy },
    { name: 'Rerank', healthy: true },
    { name: 'Context', healthy: true },
    { name: 'LLM', healthy: ollamaHealthy },
    { name: 'Response', healthy: apiHealthy },
  ];

  if (loading && !healthData) {
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
          status={healthData?.api_server.status || 'unhealthy'}
          icon={<Server size={24} />}
          details={[
            { label: 'Version', value: healthData?.version || 'Unknown' },
            { label: 'Profile', value: healthData?.profile || 'Unknown' },
            { label: 'Uptime', value: healthData ? formatUptime(healthData.uptime_seconds) : 'Unknown' },
          ]}
        />
        <HealthCard
          title="Ollama LLM"
          status={healthData?.ollama_llm.status || 'unhealthy'}
          icon={<Cpu size={24} />}
          details={[
            { label: 'Model', value: healthData?.rag_pipeline.chat_model || 'Unknown' },
            { label: 'Version', value: healthData?.ollama_llm.version || 'Unknown' },
            { label: 'Status', value: healthData?.ollama_llm.status || 'Unknown' },
          ]}
        />
        <HealthCard
          title="Vector DB"
          status={healthData?.vector_db.status || 'unhealthy'}
          icon={<Database size={24} />}
          details={[
            { label: 'Backend', value: healthData?.vector_db.name || 'Unknown' },
            { label: 'Version', value: healthData?.vector_db.version || 'Unknown' },
            { label: 'Chunks', value: healthData?.knowledge_base.total_chunks.toLocaleString() || '0' },
          ]}
        />
      </div>

      {/* Pipeline Visualization */}
      <PipelineVisualization
        stages={pipelineStages}
        embeddingModel={healthData?.rag_pipeline.embedding_model}
        chatModel={healthData?.rag_pipeline.chat_model}
        chunker={healthData?.rag_pipeline.chunker}
      />

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="space-y-3">
          <StatsCard
            title="Knowledge Base"
            icon={<HardDrive size={20} />}
            stats={[
              { label: 'Total Documents', value: healthData?.knowledge_base.total_documents || 0 },
              { label: 'Total Chunks', value: healthData?.knowledge_base.total_chunks.toLocaleString() || '0' },
              { label: 'Storage Used', value: healthData?.knowledge_base.storage_size_mb ? `${healthData.knowledge_base.storage_size_mb.toFixed(1)} MB` : 'Unknown' },
              { label: 'Profile', value: healthData?.profile || 'Unknown' },
            ]}
          />
          <Button
            variant="danger"
            icon={Trash2}
            onClick={() => setShowClearModal(true)}
            className="w-full"
            disabled={(healthData?.knowledge_base.total_chunks || 0) === 0}
          >
            Clear Knowledge Base
          </Button>
        </div>
        <StatsCard
          title="Processing Queue"
          icon={<Activity size={20} />}
          stats={[
            { label: 'Pending', value: healthData?.processing_queue.pending || 0 },
            { label: 'Processing', value: healthData?.processing_queue.processing || 0 },
            { label: 'Ready', value: healthData?.processing_queue.ready || 0 },
            { label: 'Failed', value: healthData?.processing_queue.failed || 0 },
          ]}
        />
      </div>

      {/* Clear Knowledge Base Modal */}
      <ClearKBModal
        isOpen={showClearModal}
        onClose={() => setShowClearModal(false)}
        onConfirm={handleClearKB}
        totalChunks={healthData?.knowledge_base.total_chunks || 0}
        totalDocuments={healthData?.knowledge_base.total_documents || 0}
        isLoading={clearing}
      />

    </div>
  );
}
