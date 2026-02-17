import { ArrowDown, ArrowUp, ArrowUpDown } from 'lucide-react';
import { Checkbox, Badge } from '../../ui';
import { StatusBadge } from './StatusBadge';
import { SuggestionBadge } from './SuggestionBadge';
import type { Document, Tag } from '../../../types';

interface DocumentTableProps {
  documents: Document[];
  selectedIds: Set<string>;
  onSelectionChange: (ids: Set<string>) => void;
  sortBy: string;
  sortOrder: 'asc' | 'desc';
  onSort: (field: string) => void;
  onTagClick?: (tag: Tag) => void;
  loading?: boolean;
  suggestionCounts?: Map<string, number>;
  onSuggestionClick?: (doc: Document) => void;
}

type SortableField = 'original_filename' | 'uploaded_at' | 'status';

function formatDate(dateString: string): string {
  const date = new Date(dateString);
  return date.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  });
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function DocumentTable({
  documents,
  selectedIds,
  onSelectionChange,
  sortBy,
  sortOrder,
  onSort,
  onTagClick,
  loading = false,
  suggestionCounts,
  onSuggestionClick,
}: DocumentTableProps) {
  const allSelected = documents.length > 0 && documents.every((d) => selectedIds.has(d.id));
  const someSelected = documents.some((d) => selectedIds.has(d.id)) && !allSelected;

  const handleSelectAll = () => {
    if (allSelected) {
      onSelectionChange(new Set());
    } else {
      onSelectionChange(new Set(documents.map((d) => d.id)));
    }
  };

  const handleSelectOne = (id: string) => {
    const newSet = new Set(selectedIds);
    if (newSet.has(id)) {
      newSet.delete(id);
    } else {
      newSet.add(id);
    }
    onSelectionChange(newSet);
  };

  const renderSortIcon = (field: SortableField) => {
    if (sortBy !== field) {
      return <ArrowUpDown size={14} className="text-gray-400" />;
    }
    return sortOrder === 'asc' ? (
      <ArrowUp size={14} className="text-primary" />
    ) : (
      <ArrowDown size={14} className="text-primary" />
    );
  };

  const SortableHeader = ({
    field,
    children,
  }: {
    field: SortableField;
    children: React.ReactNode;
  }) => (
    <button
      onClick={() => onSort(field)}
      className="inline-flex items-center gap-1 hover:text-primary transition-colors"
    >
      {children}
      {renderSortIcon(field)}
    </button>
  );

  if (loading) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
        <div className="p-8 text-center">
          <div className="animate-pulse space-y-4">
            <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-3/4 mx-auto" />
            <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-1/2 mx-auto" />
            <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-2/3 mx-auto" />
          </div>
        </div>
      </div>
    );
  }

  if (documents.length === 0) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
        <div className="p-8 text-center text-gray-500 dark:text-gray-400">
          No documents found
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900/50">
              <th className="w-12 px-4 py-3 text-left">
                <Checkbox
                  checked={allSelected}
                  indeterminate={someSelected}
                  onChange={handleSelectAll}
                />
              </th>
              <th className="px-4 py-3 text-left text-sm font-semibold text-gray-600 dark:text-gray-400">
                <SortableHeader field="original_filename">Filename</SortableHeader>
              </th>
              <th className="px-4 py-3 text-left text-sm font-semibold text-gray-600 dark:text-gray-400">
                Tags
              </th>
              <th className="px-4 py-3 text-left text-sm font-semibold text-gray-600 dark:text-gray-400">
                <SortableHeader field="status">Status</SortableHeader>
              </th>
              <th className="px-4 py-3 text-left text-sm font-semibold text-gray-600 dark:text-gray-400">
                Size
              </th>
              <th className="px-4 py-3 text-left text-sm font-semibold text-gray-600 dark:text-gray-400">
                <SortableHeader field="uploaded_at">Uploaded</SortableHeader>
              </th>
            </tr>
          </thead>
          <tbody>
            {documents.map((doc) => (
              <tr
                key={doc.id}
                className={`
                  border-b border-gray-100 dark:border-gray-700/50 last:border-b-0
                  hover:bg-gray-50 dark:hover:bg-gray-700/30 transition-colors
                  ${selectedIds.has(doc.id) ? 'bg-primary/5' : ''}
                `}
              >
                <td className="px-4 py-3">
                  <Checkbox
                    checked={selectedIds.has(doc.id)}
                    onChange={() => handleSelectOne(doc.id)}
                  />
                </td>
                <td className="px-4 py-3">
                  <div className="flex flex-col">
                    <span className="font-medium text-gray-900 dark:text-white truncate max-w-xs">
                      {doc.original_filename}
                    </span>
                    {doc.title && doc.title !== doc.original_filename && (
                      <span className="text-sm text-gray-500 dark:text-gray-400 truncate max-w-xs">
                        {doc.title}
                      </span>
                    )}
                  </div>
                </td>
                <td className="px-4 py-3">
                  <div className="flex flex-wrap gap-1">
                    {doc.tags.map((tag) => (
                      <button
                        key={tag.id}
                        onClick={() => onTagClick?.(tag)}
                        className="transition-transform hover:scale-105"
                      >
                        <Badge variant="primary">{tag.display_name}</Badge>
                      </button>
                    ))}
                    {doc.tags.length === 0 && (!suggestionCounts || !suggestionCounts.get(doc.id)) && (
                      <span className="text-sm text-gray-400">No tags</span>
                    )}
                    {suggestionCounts && (suggestionCounts.get(doc.id) ?? 0) > 0 && onSuggestionClick && (
                      <SuggestionBadge
                        count={suggestionCounts.get(doc.id) ?? 0}
                        onClick={() => onSuggestionClick(doc)}
                      />
                    )}
                  </div>
                </td>
                <td className="px-4 py-3">
                  <StatusBadge status={doc.status} />
                </td>
                <td className="px-4 py-3 text-sm text-gray-600 dark:text-gray-400">
                  {formatFileSize(doc.file_size)}
                </td>
                <td className="px-4 py-3 text-sm text-gray-600 dark:text-gray-400">
                  {formatDate(doc.uploaded_at)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
