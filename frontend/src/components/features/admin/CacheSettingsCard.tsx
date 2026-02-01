import { Card, Checkbox, Slider } from '../../ui';
import type { CacheSettings } from '../../../types';

interface CacheSettingsCardProps {
  settings: CacheSettings;
  onChange: (settings: CacheSettings) => void;
  disabled?: boolean;
}

export function CacheSettingsCard({
  settings,
  onChange,
  disabled = false,
}: CacheSettingsCardProps) {
  const handleChange = <K extends keyof CacheSettings>(key: K, value: CacheSettings[K]) => {
    onChange({ ...settings, [key]: value });
  };

  return (
    <Card>
      <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
        Response Cache Settings
      </h2>
      <p className="text-sm text-gray-500 dark:text-gray-400 mb-6">
        Configure caching for RAG query responses to improve performance.
      </p>

      <div className="space-y-6">
        {/* Enable Cache */}
        <Checkbox
          label="Enable Response Cache"
          checked={settings.cache_enabled}
          onChange={(e) => handleChange('cache_enabled', e.target.checked)}
          disabled={disabled}
        />
        <p className="text-xs text-gray-500 dark:text-gray-400 -mt-4 ml-6">
          Cache similar queries to reduce response time and LLM costs
        </p>

        {/* TTL */}
        <Slider
          label="Cache TTL (hours)"
          description="How long cached responses remain valid"
          value={settings.cache_ttl_hours}
          min={1}
          max={168}
          step={1}
          onChange={(e) => handleChange('cache_ttl_hours', parseInt(e.target.value, 10))}
          disabled={disabled || !settings.cache_enabled}
        />

        {/* Max Entries */}
        <Slider
          label="Maximum Cache Entries"
          description="Maximum number of queries to keep cached"
          value={settings.cache_max_entries}
          min={100}
          max={10000}
          step={100}
          onChange={(e) => handleChange('cache_max_entries', parseInt(e.target.value, 10))}
          disabled={disabled || !settings.cache_enabled}
        />

        {/* Semantic Threshold */}
        <Slider
          label="Semantic Similarity Threshold"
          description="Minimum similarity for a query to be considered a cache hit"
          value={settings.cache_semantic_threshold}
          min={0.85}
          max={0.99}
          step={0.01}
          valueFormatter={(v) => v.toFixed(2)}
          onChange={(e) => handleChange('cache_semantic_threshold', parseFloat(e.target.value))}
          disabled={disabled || !settings.cache_enabled}
        />

        {/* Min Confidence */}
        <Slider
          label="Minimum Confidence to Cache"
          description="Only cache responses with at least this confidence level"
          value={settings.cache_min_confidence}
          min={0}
          max={100}
          step={5}
          valueFormatter={(v) => `${v}%`}
          onChange={(e) => handleChange('cache_min_confidence', parseInt(e.target.value, 10))}
          disabled={disabled || !settings.cache_enabled}
        />

        {/* Auto-warm Checkbox */}
        <Checkbox
          label="Enable Auto-warm"
          checked={settings.cache_auto_warm_enabled}
          onChange={(e) => handleChange('cache_auto_warm_enabled', e.target.checked)}
          disabled={disabled || !settings.cache_enabled}
        />
        <p className="text-xs text-gray-500 dark:text-gray-400 -mt-4 ml-6">
          Automatically warm cache with frequently asked queries on startup
        </p>

        {/* Auto-warm Count */}
        <Slider
          label="Auto-warm Query Count"
          description="Number of top queries to pre-cache during auto-warm"
          value={settings.cache_auto_warm_count}
          min={5}
          max={50}
          step={5}
          onChange={(e) => handleChange('cache_auto_warm_count', parseInt(e.target.value, 10))}
          disabled={disabled || !settings.cache_enabled || !settings.cache_auto_warm_enabled}
        />
      </div>
    </Card>
  );
}
