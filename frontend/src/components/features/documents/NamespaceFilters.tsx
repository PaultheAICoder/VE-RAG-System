import { useState } from 'react';
import { X, ChevronDown, ChevronUp, Loader2 } from 'lucide-react';
import type { TagFacetItem } from '../../../types';

interface NamespaceFiltersProps {
  facets: Record<string, TagFacetItem[]>;
  activeNamespace: string | null;
  activeValue: string | null;
  onFilterChange: (namespace: string | null, value: string | null) => void;
  loading: boolean;
}

/** Human-readable display names for known namespaces. */
const NAMESPACE_LABELS: Record<string, string> = {
  client: 'Client',
  doctype: 'Document Type',
  stage: 'Stage',
  topic: 'Topic',
  department: 'Department',
  entity: 'Entity',
};

function getNamespaceLabel(ns: string): string {
  return NAMESPACE_LABELS[ns] || ns.charAt(0).toUpperCase() + ns.slice(1);
}

export function NamespaceFilters({
  facets,
  activeNamespace,
  activeValue,
  onFilterChange,
  loading,
}: NamespaceFiltersProps) {
  const [expandedNamespace, setExpandedNamespace] = useState<string | null>(null);

  const namespaceKeys = Object.keys(facets);

  // Nothing to render if no facets
  if (namespaceKeys.length === 0 && !loading) {
    return null;
  }

  if (loading) {
    return (
      <div className="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400">
        <Loader2 size={14} className="animate-spin" />
        Loading filters...
      </div>
    );
  }

  const handleNamespaceClick = (ns: string) => {
    if (expandedNamespace === ns) {
      setExpandedNamespace(null);
    } else {
      setExpandedNamespace(ns);
    }
  };

  const handleValueClick = (ns: string, value: string) => {
    // Toggle: clicking the active filter removes it
    if (activeNamespace === ns && activeValue === value) {
      onFilterChange(null, null);
    } else {
      onFilterChange(ns, value);
    }
  };

  const handleRemoveFilter = () => {
    onFilterChange(null, null);
    setExpandedNamespace(null);
  };

  // Count total docs per namespace
  const namespaceTotals = namespaceKeys.reduce(
    (acc, ns) => {
      acc[ns] = facets[ns].reduce((sum, item) => sum + item.count, 0);
      return acc;
    },
    {} as Record<string, number>
  );

  return (
    <div className="space-y-2">
      {/* Active filter indicator */}
      {activeNamespace && activeValue && (
        <div className="flex items-center gap-2">
          <span className="text-xs font-medium text-gray-500 dark:text-gray-400">
            Filtered by:
          </span>
          <button
            onClick={handleRemoveFilter}
            className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium bg-primary/10 text-primary hover:bg-primary/20 transition-colors"
          >
            {getNamespaceLabel(activeNamespace)}: {activeValue}
            <X size={12} />
          </button>
        </div>
      )}

      {/* Namespace chips */}
      <div className="flex flex-wrap gap-2">
        {namespaceKeys.map((ns) => {
          const isExpanded = expandedNamespace === ns;
          const isActive = activeNamespace === ns;
          const Icon = isExpanded ? ChevronUp : ChevronDown;

          return (
            <button
              key={ns}
              onClick={() => handleNamespaceClick(ns)}
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors border ${
                isActive
                  ? 'bg-primary/10 border-primary/30 text-primary'
                  : 'bg-gray-50 dark:bg-gray-800 border-gray-200 dark:border-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
              }`}
            >
              {getNamespaceLabel(ns)}
              <span className="text-[10px] opacity-60">({namespaceTotals[ns]})</span>
              <Icon size={12} />
            </button>
          );
        })}
      </div>

      {/* Expanded namespace values */}
      {expandedNamespace && facets[expandedNamespace] && (
        <div className="flex flex-wrap gap-1.5 pl-2 border-l-2 border-gray-200 dark:border-gray-700">
          {facets[expandedNamespace].map((item) => {
            const isItemActive =
              activeNamespace === expandedNamespace && activeValue === item.name.split(':')[1];

            return (
              <button
                key={item.name}
                onClick={() => handleValueClick(expandedNamespace, item.name.split(':')[1] || item.name)}
                className={`inline-flex items-center gap-1 px-2 py-1 rounded-md text-xs transition-colors ${
                  isItemActive
                    ? 'bg-primary text-white'
                    : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                }`}
              >
                {item.display}
                <span className="text-[10px] opacity-70">({item.count})</span>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
