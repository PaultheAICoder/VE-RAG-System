import { useState, useEffect, useCallback } from 'react';
import { Activity, RefreshCw } from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { Card, Button } from '../../ui';
import { MetricScoreCard } from './MetricScoreCard';
import { getLiveStats } from '../../../api/evaluations';
import type { LiveStats } from '../../../types';

const AUTO_REFRESH_MS = 30_000;

export function LiveQualityChart() {
  const [stats, setStats] = useState<LiveStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchStats = useCallback(async () => {
    try {
      const data = await getLiveStats();
      setStats(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load live stats');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchStats();
    const interval = setInterval(fetchStats, AUTO_REFRESH_MS);
    return () => clearInterval(interval);
  }, [fetchStats]);

  const chartData = (stats?.hourly_breakdown ?? []).map((h) => ({
    hour: h.hour.slice(11, 16), // "HH:MM" from ISO
    faithfulness: h.avg_faithfulness !== null ? +(h.avg_faithfulness * 100).toFixed(1) : null,
    relevancy: h.avg_answer_relevancy !== null ? +(h.avg_answer_relevancy * 100).toFixed(1) : null,
    count: h.count,
  }));

  if (loading) {
    return (
      <div className="py-12 text-center text-gray-500">Loading live monitoring data...</div>
    );
  }

  if (error) {
    return (
      <Card>
        <div className="text-center py-8">
          <p className="text-red-500 mb-2">{error}</p>
          <Button variant="outline" icon={RefreshCw} onClick={fetchStats}>
            Retry
          </Button>
        </div>
      </Card>
    );
  }

  if (!stats) return null;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Activity className="w-5 h-5 text-primary" />
          <h2 className="text-xl font-heading font-bold text-gray-900 dark:text-white">
            Live Monitoring
          </h2>
        </div>
        <Button variant="ghost" icon={RefreshCw} onClick={fetchStats} title="Refresh" />
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricScoreCard label="Avg Faithfulness" value={stats.avg_faithfulness} />
        <MetricScoreCard label="Avg Relevancy" value={stats.avg_answer_relevancy} />
        <div className="rounded-lg p-3 bg-gray-50 dark:bg-gray-800">
          <span className="text-xs font-medium text-gray-500 dark:text-gray-400">Scores (24h)</span>
          <div className="font-bold mt-1 text-2xl text-gray-900 dark:text-white">
            {stats.scores_last_24h}
          </div>
        </div>
        <div className="rounded-lg p-3 bg-gray-50 dark:bg-gray-800">
          <span className="text-xs font-medium text-gray-500 dark:text-gray-400">Queue</span>
          <div className="font-bold mt-1 text-2xl text-gray-900 dark:text-white">
            {stats.queue_depth}/{stats.queue_capacity}
          </div>
          {stats.drops_since_startup > 0 && (
            <span className="text-xs text-red-500">{stats.drops_since_startup} drops</span>
          )}
        </div>
      </div>

      {/* Hourly chart */}
      {chartData.length > 0 ? (
        <Card>
          <h3 className="font-medium text-gray-900 dark:text-white mb-4">Hourly Score Breakdown</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
              <XAxis dataKey="hour" tick={{ fontSize: 12 }} stroke="#9CA3AF" />
              <YAxis domain={[0, 100]} tick={{ fontSize: 12 }} stroke="#9CA3AF" unit="%" />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1F2937',
                  border: 'none',
                  borderRadius: '8px',
                  color: '#F3F4F6',
                }}
                formatter={(value: unknown, name?: string) => [
                  value != null ? `${value}%` : 'N/A',
                  name === 'faithfulness' ? 'Faithfulness' : 'Relevancy',
                ]}
                labelFormatter={(label) => `Hour: ${label}`}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="faithfulness"
                stroke="#3B82F6"
                strokeWidth={2}
                dot={{ r: 3 }}
                connectNulls
                name="Faithfulness"
              />
              <Line
                type="monotone"
                dataKey="relevancy"
                stroke="#10B981"
                strokeWidth={2}
                dot={{ r: 3 }}
                connectNulls
                name="Relevancy"
              />
            </LineChart>
          </ResponsiveContainer>
        </Card>
      ) : (
        <Card>
          <div className="text-center py-8 text-gray-500">
            No live monitoring data yet. Scores will appear as queries are sampled.
          </div>
        </Card>
      )}
    </div>
  );
}
