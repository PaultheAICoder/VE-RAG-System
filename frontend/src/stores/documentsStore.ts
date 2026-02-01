import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import type { DocumentStatus } from '../types';

const DEFAULT_PAGE_SIZE = 20;

interface DocumentsState {
  // Persisted filter state
  search: string;
  selectedTagId: string | null;
  status: DocumentStatus | null;

  // Persisted sort state
  sortBy: string;
  sortOrder: 'asc' | 'desc';

  // Persisted pagination state
  page: number;
  pageSize: number;

  // Actions
  setSearch: (search: string) => void;
  setTagFilter: (tagId: string | null) => void;
  setStatusFilter: (status: DocumentStatus | null) => void;
  setSort: (sortBy: string, sortOrder: 'asc' | 'desc') => void;
  setPage: (page: number) => void;
  resetFilters: () => void;
  validateTagFilter: (availableTagIds: string[]) => void;
  clampPage: (totalItems: number) => void;
}

export const useDocumentsStore = create<DocumentsState>()(
  persist(
    (set, get) => ({
      // Default state
      search: '',
      selectedTagId: null,
      status: null,
      sortBy: 'uploaded_at',
      sortOrder: 'desc',
      page: 1,
      pageSize: DEFAULT_PAGE_SIZE,

      // Actions
      setSearch: (search: string) => {
        set({ search, page: 1 });
      },

      setTagFilter: (tagId: string | null) => {
        set({ selectedTagId: tagId, page: 1 });
      },

      setStatusFilter: (status: DocumentStatus | null) => {
        set({ status, page: 1 });
      },

      setSort: (sortBy: string, sortOrder: 'asc' | 'desc') => {
        set({ sortBy, sortOrder });
      },

      setPage: (page: number) => {
        set({ page });
      },

      resetFilters: () => {
        set({
          search: '',
          selectedTagId: null,
          status: null,
          sortBy: 'uploaded_at',
          sortOrder: 'desc',
          page: 1,
        });
      },

      validateTagFilter: (availableTagIds: string[]) => {
        const { selectedTagId } = get();
        if (selectedTagId !== null && !availableTagIds.includes(selectedTagId)) {
          set({ selectedTagId: null });
        }
      },

      clampPage: (totalItems: number) => {
        const { page, pageSize } = get();
        const maxPage = Math.max(1, Math.ceil(totalItems / pageSize));
        if (page > maxPage) {
          set({ page: maxPage });
        }
      },
    }),
    {
      name: 'documents-view-storage',
      storage: createJSONStorage(() => sessionStorage),
      partialize: (state) => ({
        search: state.search,
        selectedTagId: state.selectedTagId,
        status: state.status,
        sortBy: state.sortBy,
        sortOrder: state.sortOrder,
        page: state.page,
        // pageSize is constant, no need to persist
      }),
    }
  )
);
