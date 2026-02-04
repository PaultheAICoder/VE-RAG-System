import { useState, useMemo, useCallback } from 'react';
import { Search, Plus, MessageSquare, Trash2 } from 'lucide-react';
import type { ChatSession } from '../../../types';
import { Input, Button, Checkbox } from '../../ui';
import { SessionItem } from './SessionItem';
import { ConfirmModal } from '../admin';
import { deleteSession, bulkDeleteSessions } from '../../../api/chat';
import { useChatStore } from '../../../stores/chatStore';

interface SessionSidebarProps {
  sessions: ChatSession[];
  activeSessionId: string | null;
  onSelectSession: (sessionId: string) => void;
  onNewSession: () => void;
  isLoading?: boolean;
  /** Enable admin features (delete, bulk selection) */
  isAdmin?: boolean;
}

export function SessionSidebar({
  sessions,
  activeSessionId,
  onSelectSession,
  onNewSession,
  isLoading = false,
  isAdmin = false,
}: SessionSidebarProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedSessions, setSelectedSessions] = useState<Set<string>>(new Set());
  const [deleteModalOpen, setDeleteModalOpen] = useState(false);
  const [sessionToDelete, setSessionToDelete] = useState<string | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  const { removeSession, removeSessions } = useChatStore();

  // Filter sessions by search query (client-side)
  const filteredSessions = useMemo(() => {
    if (!searchQuery.trim()) return sessions;

    const query = searchQuery.toLowerCase();
    return sessions.filter(
      (session) =>
        (session.title?.toLowerCase().includes(query)) ||
        (session.last_message_preview?.toLowerCase().includes(query))
    );
  }, [sessions, searchQuery]);

  // Selection helpers
  const isAllSelected = filteredSessions.length > 0 && filteredSessions.every(s => selectedSessions.has(s.id));
  const isSomeSelected = filteredSessions.some(s => selectedSessions.has(s.id));
  const selectedCount = Array.from(selectedSessions).filter(id =>
    filteredSessions.some(s => s.id === id)
  ).length;

  const handleSelectAll = useCallback(() => {
    if (isAllSelected) {
      // Deselect all
      setSelectedSessions(new Set());
    } else {
      // Select all filtered sessions
      setSelectedSessions(new Set(filteredSessions.map(s => s.id)));
    }
  }, [isAllSelected, filteredSessions]);

  const handleSelectSession = useCallback((sessionId: string, selected: boolean) => {
    setSelectedSessions(prev => {
      const newSet = new Set(prev);
      if (selected) {
        newSet.add(sessionId);
      } else {
        newSet.delete(sessionId);
      }
      return newSet;
    });
  }, []);

  // Single delete handlers
  const handleDeleteClick = useCallback((sessionId: string) => {
    setSessionToDelete(sessionId);
    setDeleteModalOpen(true);
    setDeleteError(null);
  }, []);

  // Bulk delete handlers
  const handleBulkDeleteClick = useCallback(() => {
    setSessionToDelete(null); // null indicates bulk delete
    setDeleteModalOpen(true);
    setDeleteError(null);
  }, []);

  const handleConfirmDelete = useCallback(async () => {
    setIsDeleting(true);
    setDeleteError(null);

    try {
      if (sessionToDelete) {
        // Single delete
        await deleteSession(sessionToDelete);
        removeSession(sessionToDelete);
      } else {
        // Bulk delete
        const idsToDelete = Array.from(selectedSessions);
        const response = await bulkDeleteSessions(idsToDelete);

        // Remove successfully deleted sessions from store
        const deletedIds = idsToDelete.filter(id => !response.failed_ids.includes(id));
        removeSessions(deletedIds);

        // Clear selection
        setSelectedSessions(new Set());

        // If some failed, show error
        if (response.failed_ids.length > 0) {
          setDeleteError(`${response.failed_ids.length} session(s) could not be deleted`);
          setIsDeleting(false);
          return; // Keep modal open to show error
        }
      }

      setDeleteModalOpen(false);
      setSessionToDelete(null);
    } catch (error) {
      console.error('Failed to delete session(s):', error);
      setDeleteError(error instanceof Error ? error.message : 'Failed to delete session(s)');
    } finally {
      setIsDeleting(false);
    }
  }, [sessionToDelete, selectedSessions, removeSession, removeSessions]);

  const handleCloseModal = useCallback(() => {
    if (!isDeleting) {
      setDeleteModalOpen(false);
      setSessionToDelete(null);
      setDeleteError(null);
    }
  }, [isDeleting]);

  // Get delete modal content
  const getDeleteModalContent = () => {
    if (sessionToDelete) {
      const session = sessions.find(s => s.id === sessionToDelete);
      const messageCount = session?.message_count || 0;
      return {
        title: 'Delete Chat Session',
        message: `Are you sure you want to permanently delete this chat session${messageCount > 0 ? ` and its ${messageCount} message${messageCount !== 1 ? 's' : ''}` : ''}? This action cannot be undone.`,
      };
    } else {
      const totalMessages = Array.from(selectedSessions)
        .map(id => sessions.find(s => s.id === id)?.message_count || 0)
        .reduce((a, b) => a + b, 0);
      return {
        title: `Delete ${selectedCount} Chat Session${selectedCount !== 1 ? 's' : ''}`,
        message: `Are you sure you want to permanently delete ${selectedCount} chat session${selectedCount !== 1 ? 's' : ''}${totalMessages > 0 ? ` and their ${totalMessages} message${totalMessages !== 1 ? 's' : ''}` : ''}? This action cannot be undone.`,
      };
    }
  };

  const modalContent = getDeleteModalContent();

  return (
    <div className="w-60 flex-shrink-0 bg-gray-50 dark:bg-gray-900 border-r border-gray-200 dark:border-gray-800 flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-gray-200 dark:border-gray-800">
        <div className="flex items-center justify-between mb-3">
          <h2 className="font-semibold text-gray-900 dark:text-white">Sessions</h2>
          <Button
            variant="ghost"
            size="sm"
            icon={Plus}
            onClick={onNewSession}
            title="New chat session"
          >
            New
          </Button>
        </div>

        {/* Search */}
        <Input
          icon={Search}
          placeholder="Search sessions..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="text-sm"
        />
      </div>

      {/* Bulk actions bar (admin only) */}
      {isAdmin && selectedCount > 0 && (
        <div className="px-4 py-2 border-b border-gray-200 dark:border-gray-800 bg-gray-100 dark:bg-gray-800/50">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-gray-600 dark:text-gray-400">
              {selectedCount} selected
            </span>
            <Button
              variant="danger"
              size="sm"
              icon={Trash2}
              onClick={handleBulkDeleteClick}
            >
              Delete
            </Button>
          </div>
        </div>
      )}

      {/* Select all checkbox (admin only, when sessions exist) */}
      {isAdmin && filteredSessions.length > 0 && !isLoading && (
        <div className="px-4 py-2 border-b border-gray-200 dark:border-gray-800">
          <Checkbox
            label="Select all"
            checked={isAllSelected}
            indeterminate={isSomeSelected && !isAllSelected}
            onChange={handleSelectAll}
            className="text-xs"
          />
        </div>
      )}

      {/* Session list */}
      <div className="flex-1 overflow-y-auto p-2">
        {isLoading ? (
          // Loading skeleton
          <div className="space-y-2">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="animate-pulse p-3 rounded-lg">
                <div className="flex gap-3">
                  <div className="w-6 h-6 bg-gray-200 dark:bg-gray-700 rounded" />
                  <div className="flex-1 space-y-2">
                    <div className="h-4 w-3/4 bg-gray-200 dark:bg-gray-700 rounded" />
                    <div className="h-3 w-full bg-gray-200 dark:bg-gray-700 rounded" />
                    <div className="h-3 w-1/2 bg-gray-200 dark:bg-gray-700 rounded" />
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : filteredSessions.length === 0 ? (
          // Empty state
          <div className="flex flex-col items-center justify-center py-8 text-gray-500 dark:text-gray-400">
            <MessageSquare size={32} className="mb-2 opacity-50" />
            <p className="text-sm text-center">
              {searchQuery ? 'No matching sessions' : 'No sessions yet'}
            </p>
            {!searchQuery && (
              <Button
                variant="ghost"
                size="sm"
                icon={Plus}
                onClick={onNewSession}
                className="mt-2"
              >
                Start a chat
              </Button>
            )}
          </div>
        ) : (
          // Session list
          <div className="space-y-1">
            {filteredSessions.map((session) => (
              <SessionItem
                key={session.id}
                session={session}
                isActive={session.id === activeSessionId}
                onClick={() => onSelectSession(session.id)}
                showDelete={isAdmin}
                onDelete={handleDeleteClick}
                showSelect={isAdmin}
                isSelected={selectedSessions.has(session.id)}
                onSelect={handleSelectSession}
              />
            ))}
          </div>
        )}
      </div>

      {/* Delete confirmation modal */}
      <ConfirmModal
        isOpen={deleteModalOpen}
        onClose={handleCloseModal}
        onConfirm={handleConfirmDelete}
        title={modalContent.title}
        message={deleteError ? `${modalContent.message}\n\nError: ${deleteError}` : modalContent.message}
        confirmLabel="Delete"
        isLoading={isDeleting}
        variant="danger"
      />
    </div>
  );
}
