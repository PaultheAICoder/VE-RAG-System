import { useState } from 'react';
import { ChevronDown, ChevronRight, FileText } from 'lucide-react';
import type { SourceInfo } from '../../../types';

interface CitationCardProps {
  sources: SourceInfo[];
}

export function CitationCard({ sources }: CitationCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  if (!sources || sources.length === 0) {
    return null;
  }

  return (
    <div className="mt-2 border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center gap-2 px-3 py-2 text-sm text-gray-600 dark:text-gray-400 hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
      >
        {isExpanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
        <FileText size={16} />
        <span>{sources.length} source{sources.length !== 1 ? 's' : ''}</span>
      </button>

      {isExpanded && (
        <div className="border-t border-gray-200 dark:border-gray-700 divide-y divide-gray-200 dark:divide-gray-700">
          {sources.map((source, index) => (
            <div key={source.source_id || index} className="p-3">
              <div className="font-medium text-sm text-gray-900 dark:text-white">
                {source.title || 'Untitled Document'}
              </div>
              {source.snippet && (
                <p className="mt-1 text-xs text-gray-500 dark:text-gray-400 line-clamp-3">
                  {source.snippet}
                </p>
              )}
              <div className="mt-1 text-xs text-gray-400 dark:text-gray-500">
                Chunk {source.chunk_index + 1}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
