import { useState, useMemo } from 'react';
import { Search, Plus, MessageSquare } from 'lucide-react';
import type { ChatSession } from '../../../types';
import { Input, Button } from '../../ui';
import { SessionItem } from './SessionItem';

interface SessionSidebarProps {
  sessions: ChatSession[];
  activeSessionId: string | null;
  onSelectSession: (sessionId: string) => void;
  onNewSession: () => void;
  isLoading?: boolean;
}

export function SessionSidebar({
  sessions,
  activeSessionId,
  onSelectSession,
  onNewSession,
  isLoading = false,
}: SessionSidebarProps) {
  const [searchQuery, setSearchQuery] = useState('');

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
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
