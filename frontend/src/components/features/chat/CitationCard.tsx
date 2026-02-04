import { useState } from 'react';
import { ChevronDown, ChevronRight, FileText } from 'lucide-react';
import type { SourceInfo } from '../../../types';
import type { CitationMap } from '../../../utils/citationParser';

interface CitationCardProps {
  sources: SourceInfo[];
  citationMap?: CitationMap;
}

export function CitationCard({ sources, citationMap = {} }: CitationCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  if (!sources || sources.length === 0) {
    return null;
  }

  // Get footnote number for a source
  const getFootnote = (sourceId: string): number | undefined => {
    return citationMap[sourceId];
  };

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
          {sources.map((source, index) => {
            const footnote = getFootnote(source.source_id);
            return (
              <div
                key={source.source_id || index}
                id={`source-${footnote || index + 1}`}
                className="p-3 scroll-mt-4"
              >
                <div className="flex items-start gap-2">
                  {/* Footnote number badge */}
                  <span className="flex-shrink-0 inline-flex items-center justify-center w-5 h-5 text-xs font-medium bg-primary/10 text-primary rounded">
                    {footnote || index + 1}
                  </span>
                  <div className="flex-1 min-w-0">
                    <div className="font-medium text-sm text-gray-900 dark:text-white truncate">
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
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
