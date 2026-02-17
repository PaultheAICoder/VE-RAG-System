import { useState, useEffect, useCallback } from 'react';
import { RefreshCw, ChevronDown, ChevronUp, Tag, Shield } from 'lucide-react';
import { Card, Button, Badge, Alert, Select } from '../../ui';
import { ConfirmModal } from './ConfirmModal';
import {
  getAutoTagStrategies,
  getAutoTagStrategy,
  switchActiveAutoTagStrategy,
} from '../../../api/admin';
import type {
  StrategyListItem,
  StrategyDetailResponse,
} from '../../../types';

export function AutoTagStrategyCard() {
  const [strategies, setStrategies] = useState<StrategyListItem[]>([]);
  const [activeStrategyId, setActiveStrategyId] = useState<string>('');
  const [selectedDetail, setSelectedDetail] = useState<StrategyDetailResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showConfirmSwitch, setShowConfirmSwitch] = useState(false);
  const [pendingStrategyId, setPendingStrategyId] = useState<string>('');
  const [switching, setSwitching] = useState(false);
  const [previewOpen, setPreviewOpen] = useState(true);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getAutoTagStrategies();
      setStrategies(data.strategies);
      setActiveStrategyId(data.active_strategy_id);
      // Fetch detail for active strategy
      if (data.active_strategy_id) {
        const detail = await getAutoTagStrategy(data.active_strategy_id);
        setSelectedDetail(detail);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load strategies');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const handleStrategyChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newId = e.target.value;
    if (newId !== activeStrategyId) {
      setPendingStrategyId(newId);
      setShowConfirmSwitch(true);
    }
  };

  const confirmSwitch = async () => {
    setShowConfirmSwitch(false);
    setSwitching(true);
    setError(null);
    try {
      await switchActiveAutoTagStrategy(pendingStrategyId);
      setActiveStrategyId(pendingStrategyId);
      const detail = await getAutoTagStrategy(pendingStrategyId);
      setSelectedDetail(detail);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to switch strategy');
    } finally {
      setSwitching(false);
      setPendingStrategyId('');
    }
  };

  const cancelSwitch = () => {
    setShowConfirmSwitch(false);
    setPendingStrategyId('');
  };

  const activeStrategy = strategies.find((s) => s.id === activeStrategyId);
  const pendingStrategy = strategies.find((s) => s.id === pendingStrategyId);

  const strategyOptions = strategies.map((s) => ({
    value: s.id,
    label: `${s.name} (v${s.version})`,
  }));

  return (
    <Card>
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
            Auto-Tagging Strategy
          </h2>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            Select which tagging strategy is used for new document uploads.
          </p>
        </div>
        <Button
          variant="secondary"
          size="sm"
          icon={RefreshCw}
          onClick={fetchData}
          disabled={loading}
        >
          {loading ? 'Loading...' : 'Refresh'}
        </Button>
      </div>

      {error && (
        <Alert variant="danger" onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {loading && strategies.length === 0 && (
        <p className="text-gray-500 dark:text-gray-400 text-sm">Loading strategies...</p>
      )}

      {!loading && strategies.length === 0 && !error && (
        <p className="text-gray-500 dark:text-gray-400 text-sm">
          No strategies found. Check backend configuration.
        </p>
      )}

      {strategies.length > 0 && (
        <div className="space-y-4">
          {/* Strategy Selector */}
          <div className="flex items-end gap-4">
            <div className="flex-1">
              <Select
                label="Active Strategy"
                value={activeStrategyId}
                onChange={handleStrategyChange}
                disabled={switching}
                options={strategyOptions}
              />
            </div>
            {activeStrategy && (
              <div className="flex items-center gap-2 pb-1">
                <Badge variant="primary">v{activeStrategy.version}</Badge>
                {activeStrategy.is_builtin && (
                  <Badge variant="default">
                    <span className="flex items-center gap-1">
                      <Shield size={12} />
                      Built-in
                    </span>
                  </Badge>
                )}
              </div>
            )}
          </div>

          {switching && (
            <p className="text-sm text-primary">Switching strategy...</p>
          )}

          {/* Active Strategy Summary */}
          {activeStrategy && (
            <div className="grid grid-cols-3 gap-4 text-center p-3 bg-gray-50 dark:bg-gray-800/50 rounded-lg">
              <div>
                <div className="text-xl font-bold text-gray-900 dark:text-white">
                  {activeStrategy.namespace_count}
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-400">Namespaces</div>
              </div>
              <div>
                <div className="text-xl font-bold text-gray-900 dark:text-white">
                  {activeStrategy.document_type_count}
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-400">Document Types</div>
              </div>
              <div>
                <div className="text-xl font-bold text-gray-900 dark:text-white">
                  {activeStrategy.path_rule_count}
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-400">Path Rules</div>
              </div>
            </div>
          )}

          {/* Strategy Preview */}
          {selectedDetail && (
            <div className="border border-gray-200 dark:border-gray-700 rounded-lg">
              <button
                onClick={() => setPreviewOpen(!previewOpen)}
                className="w-full flex items-center justify-between p-3 text-sm font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-800/50 rounded-lg"
              >
                <span>Strategy Preview</span>
                {previewOpen ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
              </button>

              {previewOpen && (
                <div className="px-3 pb-3 space-y-4">
                  {/* Description */}
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    {selectedDetail.description}
                  </p>

                  {/* Namespaces */}
                  <div>
                    <h4 className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-2">
                      Namespaces
                    </h4>
                    <div className="flex flex-wrap gap-2">
                      {Object.entries(selectedDetail.namespaces).map(([key, ns]) => (
                        <span
                          key={key}
                          className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300"
                        >
                          <span
                            className="w-2.5 h-2.5 rounded-full"
                            style={{ backgroundColor: ns.color }}
                          />
                          {ns.display}
                        </span>
                      ))}
                    </div>
                  </div>

                  {/* Document Types */}
                  <div>
                    <h4 className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-2">
                      Document Types
                    </h4>
                    <div className="flex flex-wrap gap-2">
                      {Object.entries(selectedDetail.document_types).map(([key, dt]) => (
                        <Badge key={key} variant="default">
                          <span className="flex items-center gap-1">
                            <Tag size={10} />
                            {dt.display}
                          </span>
                        </Badge>
                      ))}
                    </div>
                  </div>

                  {/* Path Rules */}
                  {selectedDetail.path_rules.length > 0 && (
                    <div>
                      <h4 className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-2">
                        Path Rules
                      </h4>
                      <div className="space-y-1">
                        {selectedDetail.path_rules.map((rule, idx) => (
                          <div
                            key={idx}
                            className="flex items-center gap-2 text-xs text-gray-600 dark:text-gray-400"
                          >
                            <Badge variant="primary">{rule.namespace}</Badge>
                            {rule.pattern && (
                              <code className="px-1.5 py-0.5 bg-gray-100 dark:bg-gray-700 rounded text-xs">
                                {rule.pattern}
                              </code>
                            )}
                            {rule.level !== undefined && (
                              <span>level: {rule.level}</span>
                            )}
                            {rule.transform && (
                              <span className="text-gray-400">({rule.transform})</span>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Extraction Config */}
                  <div className="flex gap-4">
                    <div className="flex items-center gap-2 text-xs">
                      <span className="text-gray-500 dark:text-gray-400">Entity Extraction:</span>
                      <Badge variant={selectedDetail.entity_extraction ? 'success' : 'default'}>
                        {selectedDetail.entity_extraction ? 'Enabled' : 'Disabled'}
                      </Badge>
                    </div>
                    <div className="flex items-center gap-2 text-xs">
                      <span className="text-gray-500 dark:text-gray-400">Topic Extraction:</span>
                      <Badge variant={selectedDetail.topic_extraction ? 'success' : 'default'}>
                        {selectedDetail.topic_extraction ? 'Enabled' : 'Disabled'}
                      </Badge>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Confirm Strategy Switch Modal */}
      <ConfirmModal
        isOpen={showConfirmSwitch}
        onClose={cancelSwitch}
        onConfirm={confirmSwitch}
        title="Switch Auto-Tagging Strategy"
        message={`Switch from "${activeStrategy?.name || 'current'}" to "${pendingStrategy?.name || 'selected'}"? New uploads will use the new strategy. Existing document tags are not affected.`}
        confirmLabel="Yes, Switch Strategy"
        variant="warning"
        isLoading={switching}
      />
    </Card>
  );
}
