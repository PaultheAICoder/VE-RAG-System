import { useState } from 'react';
import { SynonymManager, QAManager, EvaluationManager } from '../components/features/admin';

type TabType = 'synonyms' | 'qa' | 'evaluations';

interface Tab {
  id: TabType;
  label: string;
}

const tabs: Tab[] = [
  { id: 'synonyms', label: 'Synonyms' },
  { id: 'qa', label: 'Curated Q&A' },
  { id: 'evaluations', label: 'Evaluations' },
];

export function RAGQualityView() {
  const [activeTab, setActiveTab] = useState<TabType>('synonyms');

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-heading font-bold text-gray-900 dark:text-white">
          RAG Quality
        </h1>
        <p className="text-gray-500 dark:text-gray-400 mt-1">
          Improve RAG performance with synonyms, curated Q&A, and evaluations
        </p>
      </div>

      {/* Tab Navigation */}
      <div className="border-b border-gray-200 dark:border-gray-700">
        <nav className="flex gap-4" aria-label="Tabs">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`
                py-3 px-1 text-sm font-medium border-b-2 -mb-px transition-colors
                ${
                  activeTab === tab.id
                    ? 'border-primary text-primary'
                    : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 hover:border-gray-300 dark:hover:border-gray-600'
                }
              `}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      {activeTab === 'synonyms' && <SynonymManager />}
      {activeTab === 'qa' && <QAManager />}
      {activeTab === 'evaluations' && <EvaluationManager />}
    </div>
  );
}
