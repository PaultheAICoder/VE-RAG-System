import { useState, useEffect, useCallback } from 'react';
import { Save, AlertTriangle, RefreshCw } from 'lucide-react';
import { Button, Alert, Card, Select, Checkbox, Slider } from '../components/ui';
import { ConfirmModal } from '../components/features/admin';
import {
  getProcessingOptions,
  updateProcessingOptions,
  getModels,
  changeChatModel,
  changeEmbeddingModel,
  getRetrievalSettings,
  updateRetrievalSettings,
  getLLMSettings,
  updateLLMSettings,
} from '../api/admin';
import type { ProcessingOptions, ModelsResponse, RetrievalSettings, LLMSettings } from '../types';

const OCR_LANGUAGE_OPTIONS = [
  { value: 'eng', label: 'English' },
  { value: 'deu', label: 'German' },
  { value: 'fra', label: 'French' },
  { value: 'spa', label: 'Spanish' },
  { value: 'ita', label: 'Italian' },
  { value: 'por', label: 'Portuguese' },
  { value: 'nld', label: 'Dutch' },
  { value: 'pol', label: 'Polish' },
  { value: 'rus', label: 'Russian' },
  { value: 'jpn', label: 'Japanese' },
  { value: 'chi_sim', label: 'Chinese (Simplified)' },
  { value: 'kor', label: 'Korean' },
];

const TABLE_MODE_OPTIONS = [
  { value: 'accurate', label: 'Accurate (slower)' },
  { value: 'fast', label: 'Fast (less precise)' },
];

export function SettingsView() {
  // Data state
  const [options, setOptions] = useState<ProcessingOptions | null>(null);
  const [models, setModels] = useState<ModelsResponse | null>(null);
  const [retrievalSettings, setRetrievalSettings] = useState<RetrievalSettings | null>(null);
  const [llmSettings, setLlmSettings] = useState<LLMSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // Form state
  const [formOptions, setFormOptions] = useState<ProcessingOptions | null>(null);
  const [selectedChatModel, setSelectedChatModel] = useState('');
  const [selectedEmbeddingModel, setSelectedEmbeddingModel] = useState('');
  const [formRetrievalSettings, setFormRetrievalSettings] = useState<RetrievalSettings | null>(
    null
  );
  const [formLlmSettings, setFormLlmSettings] = useState<LLMSettings | null>(null);
  const [isDirty, setIsDirty] = useState(false);

  // Modal state
  const [showEmbeddingWarning, setShowEmbeddingWarning] = useState(false);
  const [pendingEmbeddingModel, setPendingEmbeddingModel] = useState('');
  const [saving, setSaving] = useState(false);

  // Fetch data
  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [optionsData, modelsData, retrievalData, llmData] = await Promise.all([
        getProcessingOptions(),
        getModels(),
        getRetrievalSettings(),
        getLLMSettings(),
      ]);
      setOptions(optionsData);
      setFormOptions(optionsData);
      setModels(modelsData);
      setSelectedChatModel(modelsData.current_chat_model);
      setSelectedEmbeddingModel(modelsData.current_embedding_model);
      setRetrievalSettings(retrievalData);
      setFormRetrievalSettings(retrievalData);
      setLlmSettings(llmData);
      setFormLlmSettings(llmData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load settings');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Track dirty state
  useEffect(() => {
    if (!options || !formOptions || !models) return;

    const optionsDirty =
      formOptions.enable_ocr !== options.enable_ocr ||
      formOptions.force_full_page_ocr !== options.force_full_page_ocr ||
      formOptions.ocr_language !== options.ocr_language ||
      formOptions.table_extraction_mode !== options.table_extraction_mode ||
      formOptions.include_image_descriptions !== options.include_image_descriptions ||
      formOptions.query_routing_mode !== options.query_routing_mode;

    const modelsDirty =
      selectedChatModel !== models.current_chat_model ||
      selectedEmbeddingModel !== models.current_embedding_model;

    const retrievalDirty =
      formRetrievalSettings &&
      retrievalSettings &&
      (formRetrievalSettings.retrieval_top_k !== retrievalSettings.retrieval_top_k ||
        formRetrievalSettings.retrieval_min_score !== retrievalSettings.retrieval_min_score ||
        formRetrievalSettings.retrieval_enable_expansion !==
          retrievalSettings.retrieval_enable_expansion);

    const llmDirty =
      formLlmSettings &&
      llmSettings &&
      (formLlmSettings.llm_temperature !== llmSettings.llm_temperature ||
        formLlmSettings.llm_max_response_tokens !== llmSettings.llm_max_response_tokens ||
        formLlmSettings.llm_confidence_threshold !== llmSettings.llm_confidence_threshold);

    setIsDirty(optionsDirty || modelsDirty || !!retrievalDirty || !!llmDirty);
  }, [
    formOptions,
    options,
    selectedChatModel,
    selectedEmbeddingModel,
    models,
    formRetrievalSettings,
    retrievalSettings,
    formLlmSettings,
    llmSettings,
  ]);

  // Handle form changes
  const handleOptionChange = (key: keyof ProcessingOptions, value: boolean | string) => {
    if (!formOptions) return;
    setFormOptions({ ...formOptions, [key]: value });
  };

  // Handle embedding model change (with warning)
  const handleEmbeddingModelChange = (value: string) => {
    if (value !== models?.current_embedding_model) {
      setPendingEmbeddingModel(value);
      setShowEmbeddingWarning(true);
    } else {
      setSelectedEmbeddingModel(value);
    }
  };

  const confirmEmbeddingChange = () => {
    setSelectedEmbeddingModel(pendingEmbeddingModel);
    setShowEmbeddingWarning(false);
    setPendingEmbeddingModel('');
  };

  // Handle save
  const handleSave = async () => {
    if (!formOptions || !models) return;
    setSaving(true);
    setError(null);
    setSuccess(null);

    try {
      // Save processing options if changed
      const optionsDirty =
        formOptions.enable_ocr !== options?.enable_ocr ||
        formOptions.force_full_page_ocr !== options?.force_full_page_ocr ||
        formOptions.ocr_language !== options?.ocr_language ||
        formOptions.table_extraction_mode !== options?.table_extraction_mode ||
        formOptions.include_image_descriptions !== options?.include_image_descriptions ||
        formOptions.query_routing_mode !== options?.query_routing_mode;

      if (optionsDirty) {
        await updateProcessingOptions(formOptions);
      }

      // Change chat model if different
      if (selectedChatModel !== models.current_chat_model) {
        await changeChatModel(selectedChatModel);
      }

      // Change embedding model if different (requires confirmation already done)
      if (selectedEmbeddingModel !== models.current_embedding_model) {
        await changeEmbeddingModel(selectedEmbeddingModel, true);
      }

      // Save retrieval settings if changed
      if (
        formRetrievalSettings &&
        retrievalSettings &&
        (formRetrievalSettings.retrieval_top_k !== retrievalSettings.retrieval_top_k ||
          formRetrievalSettings.retrieval_min_score !== retrievalSettings.retrieval_min_score ||
          formRetrievalSettings.retrieval_enable_expansion !==
            retrievalSettings.retrieval_enable_expansion)
      ) {
        await updateRetrievalSettings(formRetrievalSettings);
      }

      // Save LLM settings if changed
      if (
        formLlmSettings &&
        llmSettings &&
        (formLlmSettings.llm_temperature !== llmSettings.llm_temperature ||
          formLlmSettings.llm_max_response_tokens !== llmSettings.llm_max_response_tokens ||
          formLlmSettings.llm_confidence_threshold !== llmSettings.llm_confidence_threshold)
      ) {
        await updateLLMSettings(formLlmSettings);
      }

      setSuccess('Settings saved successfully');
      await fetchData(); // Refresh to get updated state
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save settings');
    } finally {
      setSaving(false);
    }
  };

  // Handle cancel
  const handleCancel = () => {
    if (options) setFormOptions(options);
    if (models) {
      setSelectedChatModel(models.current_chat_model);
      setSelectedEmbeddingModel(models.current_embedding_model);
    }
    if (retrievalSettings) setFormRetrievalSettings(retrievalSettings);
    if (llmSettings) setFormLlmSettings(llmSettings);
    setIsDirty(false);
  };

  if (loading) {
    return (
      <div className="p-6 flex items-center justify-center h-64">
        <div className="text-gray-500">Loading settings...</div>
      </div>
    );
  }

  if (!formOptions || !models || !formRetrievalSettings || !formLlmSettings) {
    return (
      <div className="p-6">
        <Alert variant="danger">Failed to load settings. Please try again.</Alert>
      </div>
    );
  }

  const chatModelOptions = models.chat_models.map((m) => ({
    value: m.name,
    label: `${m.display_name} (${m.size_gb.toFixed(1)} GB)`,
  }));

  const embeddingModelOptions = models.embedding_models.map((m) => ({
    value: m.name,
    label: `${m.display_name} (${m.size_gb.toFixed(1)} GB)`,
  }));

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-heading font-bold text-gray-900 dark:text-white">
            Settings
          </h1>
          <p className="text-gray-500 dark:text-gray-400 mt-1">
            Configure system-wide options
          </p>
        </div>
        <Button icon={RefreshCw} variant="secondary" onClick={fetchData}>
          Refresh
        </Button>
      </div>

      {/* Alerts */}
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

      {/* Models Section */}
      <Card>
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Models</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Select
            label="Chat Model"
            value={selectedChatModel}
            onChange={(e) => setSelectedChatModel(e.target.value)}
            options={chatModelOptions}
          />
          <div>
            <Select
              label="Embedding Model"
              value={selectedEmbeddingModel}
              onChange={(e) => handleEmbeddingModelChange(e.target.value)}
              options={embeddingModelOptions}
            />
            {selectedEmbeddingModel !== models.current_embedding_model && (
              <div className="mt-2 flex items-center gap-2 text-amber-600 dark:text-amber-400 text-sm">
                <AlertTriangle size={16} />
                <span>Changing requires re-indexing all documents</span>
              </div>
            )}
          </div>
        </div>
      </Card>

      {/* Document Processing Section */}
      <Card>
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Document Processing
        </h2>
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <Checkbox
                label="Enable OCR"
                checked={formOptions.enable_ocr}
                onChange={(e) => handleOptionChange('enable_ocr', e.target.checked)}
              />
              <Checkbox
                label="Force full-page OCR"
                checked={formOptions.force_full_page_ocr}
                onChange={(e) => handleOptionChange('force_full_page_ocr', e.target.checked)}
              />
              <Checkbox
                label="Include image descriptions"
                checked={formOptions.include_image_descriptions}
                onChange={(e) => handleOptionChange('include_image_descriptions', e.target.checked)}
              />
            </div>
            <div className="space-y-4">
              <Select
                label="OCR Language"
                value={formOptions.ocr_language}
                onChange={(e) => handleOptionChange('ocr_language', e.target.value)}
                options={OCR_LANGUAGE_OPTIONS}
              />
              <Select
                label="Table Extraction Mode"
                value={formOptions.table_extraction_mode}
                onChange={(e) =>
                  handleOptionChange('table_extraction_mode', e.target.value as 'accurate' | 'fast')
                }
                options={TABLE_MODE_OPTIONS}
              />
            </div>
          </div>
        </div>
      </Card>

      {/* Query Routing Section */}
      <Card>
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Query Routing</h2>
        <div className="space-y-3">
          <label className="flex items-start gap-3 cursor-pointer">
            <input
              type="radio"
              name="queryRouting"
              value="retrieve_only"
              checked={formOptions.query_routing_mode === 'retrieve_only'}
              onChange={(e) => handleOptionChange('query_routing_mode', e.target.value)}
              className="mt-1"
            />
            <div>
              <span className="font-medium text-gray-900 dark:text-white">Retrieve Only</span>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Always search documents before answering (recommended)
              </p>
            </div>
          </label>
          <label className="flex items-start gap-3 cursor-pointer">
            <input
              type="radio"
              name="queryRouting"
              value="retrieve_and_direct"
              checked={formOptions.query_routing_mode === 'retrieve_and_direct'}
              onChange={(e) => handleOptionChange('query_routing_mode', e.target.value)}
              className="mt-1"
            />
            <div>
              <span className="font-medium text-gray-900 dark:text-white">Auto Route</span>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                LLM decides when to search vs answer directly
              </p>
            </div>
          </label>
        </div>
      </Card>

      {/* Retrieval Settings Section */}
      <Card>
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Retrieval Settings
        </h2>
        <p className="text-sm text-gray-500 dark:text-gray-400 mb-6">
          Control how documents are retrieved and ranked for context.
        </p>
        <div className="space-y-6">
          <Slider
            label="Results to Retrieve (top-k)"
            description="Number of document chunks to retrieve for each query"
            value={formRetrievalSettings.retrieval_top_k}
            min={3}
            max={20}
            step={1}
            onChange={(e) =>
              setFormRetrievalSettings({
                ...formRetrievalSettings,
                retrieval_top_k: parseInt(e.target.value, 10),
              })
            }
          />
          <Slider
            label="Minimum Similarity Score"
            description="Filter out chunks below this relevance threshold"
            value={formRetrievalSettings.retrieval_min_score}
            min={0.1}
            max={0.9}
            step={0.05}
            valueFormatter={(v) => v.toFixed(2)}
            onChange={(e) =>
              setFormRetrievalSettings({
                ...formRetrievalSettings,
                retrieval_min_score: parseFloat(e.target.value),
              })
            }
          />
          <Checkbox
            label="Enable Query Expansion"
            checked={formRetrievalSettings.retrieval_enable_expansion}
            onChange={(e) =>
              setFormRetrievalSettings({
                ...formRetrievalSettings,
                retrieval_enable_expansion: e.target.checked,
              })
            }
          />
          <p className="text-xs text-gray-500 dark:text-gray-400 -mt-2 ml-6">
            Automatically rephrase queries to improve retrieval quality
          </p>
        </div>
      </Card>

      {/* LLM Response Settings Section */}
      <Card>
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          LLM Response Settings
        </h2>
        <p className="text-sm text-gray-500 dark:text-gray-400 mb-6">
          Configure how the language model generates responses.
        </p>
        <div className="space-y-6">
          <Slider
            label="Temperature"
            description="Higher values make output more random, lower values more deterministic"
            value={formLlmSettings.llm_temperature}
            min={0.0}
            max={1.0}
            step={0.1}
            valueFormatter={(v) => v.toFixed(1)}
            onChange={(e) =>
              setFormLlmSettings({
                ...formLlmSettings,
                llm_temperature: parseFloat(e.target.value),
              })
            }
          />
          <Slider
            label="Max Response Tokens"
            description="Maximum length of generated responses"
            value={formLlmSettings.llm_max_response_tokens}
            min={256}
            max={4096}
            step={256}
            onChange={(e) =>
              setFormLlmSettings({
                ...formLlmSettings,
                llm_max_response_tokens: parseInt(e.target.value, 10),
              })
            }
          />
          <Slider
            label="Confidence Threshold"
            description="Below this threshold, responses will suggest consulting a human expert"
            value={formLlmSettings.llm_confidence_threshold}
            min={0}
            max={100}
            step={5}
            valueFormatter={(v) => `${v}%`}
            onChange={(e) =>
              setFormLlmSettings({
                ...formLlmSettings,
                llm_confidence_threshold: parseInt(e.target.value, 10),
              })
            }
          />
        </div>
      </Card>

      {/* Action Buttons */}
      <div className="flex justify-end gap-3">
        <Button variant="secondary" onClick={handleCancel} disabled={!isDirty || saving}>
          Cancel
        </Button>
        <Button icon={Save} onClick={handleSave} disabled={!isDirty || saving}>
          {saving ? 'Saving...' : 'Save Changes'}
        </Button>
      </div>

      {/* Embedding Change Warning Modal */}
      <ConfirmModal
        isOpen={showEmbeddingWarning}
        onClose={() => {
          setShowEmbeddingWarning(false);
          setPendingEmbeddingModel('');
        }}
        onConfirm={confirmEmbeddingChange}
        title="Change Embedding Model"
        message="Changing the embedding model will invalidate all existing vectors. You will need to re-index all documents for search to work correctly. Are you sure you want to proceed?"
        confirmLabel="Yes, Change Model"
        variant="warning"
      />
    </div>
  );
}
