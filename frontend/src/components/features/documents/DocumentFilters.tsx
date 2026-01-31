import { useState, useEffect, useCallback } from 'react';
import { Search, X } from 'lucide-react';
import { Input, Select } from '../../ui';
import type { Tag, DocumentStatus } from '../../../types';

interface DocumentFiltersProps {
  search: string;
  onSearchChange: (value: string) => void;
  selectedTagId: string | null;
  onTagChange: (tagId: string | null) => void;
  status: DocumentStatus | null;
  onStatusChange: (status: DocumentStatus | null) => void;
  tags: Tag[];
}

const STATUS_OPTIONS = [
  { value: '', label: 'All Statuses' },
  { value: 'ready', label: 'Ready' },
  { value: 'pending', label: 'Pending' },
  { value: 'processing', label: 'Processing' },
  { value: 'failed', label: 'Failed' },
];

export function DocumentFilters({
  search,
  onSearchChange,
  selectedTagId,
  onTagChange,
  status,
  onStatusChange,
  tags,
}: DocumentFiltersProps) {
  const [localSearch, setLocalSearch] = useState(search);

  // Debounce search input
  useEffect(() => {
    const timer = setTimeout(() => {
      onSearchChange(localSearch);
    }, 300);

    return () => clearTimeout(timer);
  }, [localSearch, onSearchChange]);

  // Sync external search changes
  useEffect(() => {
    setLocalSearch(search);
  }, [search]);

  const tagOptions = [
    { value: '', label: 'All Tags' },
    ...tags.map((tag) => ({ value: tag.id, label: tag.display_name })),
  ];

  const handleClearSearch = useCallback(() => {
    setLocalSearch('');
    onSearchChange('');
  }, [onSearchChange]);

  return (
    <div className="flex flex-wrap items-center gap-4">
      {/* Search Input */}
      <div className="relative flex-1 min-w-[200px] max-w-sm">
        <Input
          icon={Search}
          placeholder="Search documents..."
          value={localSearch}
          onChange={(e) => setLocalSearch(e.target.value)}
        />
        {localSearch && (
          <button
            onClick={handleClearSearch}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
          >
            <X size={16} />
          </button>
        )}
      </div>

      {/* Tag Filter */}
      <Select
        options={tagOptions}
        value={selectedTagId || ''}
        onChange={(e) => onTagChange(e.target.value || null)}
        className="w-40"
      />

      {/* Status Filter */}
      <Select
        options={STATUS_OPTIONS}
        value={status || ''}
        onChange={(e) =>
          onStatusChange((e.target.value as DocumentStatus) || null)
        }
        className="w-40"
      />
    </div>
  );
}
