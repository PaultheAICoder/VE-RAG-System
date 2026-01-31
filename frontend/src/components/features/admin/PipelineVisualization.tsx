import { Check, X, ChevronRight } from 'lucide-react';
import { Card } from '../../ui';

interface PipelineStage {
  name: string;
  healthy: boolean;
}

interface PipelineVisualizationProps {
  stages: PipelineStage[];
  embeddingModel?: string;
  chatModel?: string;
  chunker?: string;
}

export function PipelineVisualization({
  stages,
  embeddingModel,
  chatModel,
  chunker,
}: PipelineVisualizationProps) {
  return (
    <Card variant="elevated">
      <h3 className="font-semibold text-gray-900 dark:text-white mb-4">RAG Pipeline</h3>

      {/* Pipeline stages */}
      <div className="flex flex-wrap items-center gap-1 mb-4">
        {stages.map((stage, index) => (
          <div key={stage.name} className="flex items-center">
            <div className="flex flex-col items-center">
              <span className="text-xs text-gray-500 dark:text-gray-400 mb-1">{stage.name}</span>
              <div
                className={`w-8 h-8 rounded-full flex items-center justify-center ${
                  stage.healthy
                    ? 'bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400'
                    : 'bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400'
                }`}
              >
                {stage.healthy ? <Check size={16} /> : <X size={16} />}
              </div>
            </div>
            {index < stages.length - 1 && (
              <ChevronRight
                size={16}
                className="text-gray-300 dark:text-gray-600 mx-1 flex-shrink-0"
              />
            )}
          </div>
        ))}
      </div>

      {/* Configuration details */}
      <dl className="space-y-2 pt-3 border-t border-gray-200 dark:border-gray-700">
        {embeddingModel && (
          <div className="flex items-center gap-2 text-sm">
            <dt className="text-gray-500 dark:text-gray-400">Embedding Model:</dt>
            <dd className="text-gray-700 dark:text-gray-300">{embeddingModel}</dd>
          </div>
        )}
        {chatModel && (
          <div className="flex items-center gap-2 text-sm">
            <dt className="text-gray-500 dark:text-gray-400">Chat Model:</dt>
            <dd className="text-gray-700 dark:text-gray-300">{chatModel}</dd>
          </div>
        )}
        {chunker && (
          <div className="flex items-center gap-2 text-sm">
            <dt className="text-gray-500 dark:text-gray-400">Chunker:</dt>
            <dd className="text-gray-700 dark:text-gray-300">{chunker}</dd>
          </div>
        )}
      </dl>
    </Card>
  );
}
