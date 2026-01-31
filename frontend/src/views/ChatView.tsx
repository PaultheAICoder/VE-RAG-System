import { useState, useEffect, useCallback } from 'react';
import { Wifi, WifiOff, AlertCircle } from 'lucide-react';
import { apiClient } from '../api/client';
import type {
  ChatSession,
  ChatMessage,
  SessionListResponse,
  MessageListResponse,
  SendMessageResponse,
} from '../types';
import { SessionSidebar } from '../components/features/chat/SessionSidebar';
import { MessageList } from '../components/features/chat/MessageList';
import { ChatInput } from '../components/features/chat/ChatInput';
import { Alert, Badge } from '../components/ui';

export function ChatView() {
  // Sessions state
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [isLoadingSessions, setIsLoadingSessions] = useState(true);
  const [sessionsError, setSessionsError] = useState<string | null>(null);

  // Messages state
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoadingMessages, setIsLoadingMessages] = useState(false);
  const [messagesError, setMessagesError] = useState<string | null>(null);

  // Sending state
  const [isSending, setIsSending] = useState(false);
  const [typingMessage, setTypingMessage] = useState<ChatMessage | null>(null);

  // Connection status (for future WebSocket, currently shows HTTP)
  const [isConnected] = useState(true);

  // Load sessions on mount
  useEffect(() => {
    loadSessions();
  }, []);

  // Load messages when active session changes
  useEffect(() => {
    if (activeSessionId) {
      loadMessages(activeSessionId);
    } else {
      setMessages([]);
    }
  }, [activeSessionId]);

  const loadSessions = async () => {
    setIsLoadingSessions(true);
    setSessionsError(null);
    try {
      const response = await apiClient.get<SessionListResponse>('/api/chat/sessions?limit=50');
      setSessions(response.sessions);

      // Auto-select first session if none selected
      if (response.sessions.length > 0 && !activeSessionId) {
        setActiveSessionId(response.sessions[0].id);
      }
    } catch (error) {
      console.error('Failed to load sessions:', error);
      setSessionsError(error instanceof Error ? error.message : 'Failed to load sessions');
    } finally {
      setIsLoadingSessions(false);
    }
  };

  const loadMessages = async (sessionId: string) => {
    setIsLoadingMessages(true);
    setMessagesError(null);
    try {
      const response = await apiClient.get<MessageListResponse>(
        `/api/chat/sessions/${sessionId}/messages?limit=50`
      );
      setMessages(response.messages);
    } catch (error) {
      console.error('Failed to load messages:', error);
      setMessagesError(error instanceof Error ? error.message : 'Failed to load messages');
    } finally {
      setIsLoadingMessages(false);
    }
  };

  const createSession = async (): Promise<string | null> => {
    try {
      const response = await apiClient.post<ChatSession>('/api/chat/sessions', {
        title: null,
      });

      // Add new session to the top of the list
      setSessions((prev) => [
        {
          ...response,
          message_count: 0,
          last_message_preview: undefined,
        },
        ...prev,
      ]);

      return response.id;
    } catch (error) {
      console.error('Failed to create session:', error);
      setSessionsError(error instanceof Error ? error.message : 'Failed to create session');
      return null;
    }
  };

  const handleNewSession = useCallback(async () => {
    const sessionId = await createSession();
    if (sessionId) {
      setActiveSessionId(sessionId);
      setMessages([]);
    }
  }, []);

  // Handle Cmd/Ctrl + N keyboard shortcut for new session in chat view
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const isMod = e.metaKey || e.ctrlKey;
      const target = e.target as HTMLElement;
      const tagName = target.tagName.toLowerCase();
      const isInputField =
        tagName === 'input' ||
        tagName === 'textarea' ||
        target.isContentEditable;

      // Cmd/Ctrl + N: New session (only when not in input field)
      if (isMod && e.key === 'n' && !isInputField) {
        e.preventDefault();
        handleNewSession();
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [handleNewSession]);

  const handleSelectSession = useCallback((sessionId: string) => {
    setActiveSessionId(sessionId);
  }, []);

  const handleSendMessage = async (content: string) => {
    if (!content.trim() || isSending) return;

    let sessionId = activeSessionId;

    // Create a new session if none exists
    if (!sessionId) {
      sessionId = await createSession();
      if (!sessionId) return;
      setActiveSessionId(sessionId);
    }

    // Create optimistic user message
    const optimisticUserMessage: ChatMessage = {
      id: `temp-${Date.now()}`,
      role: 'user',
      content,
      created_at: new Date().toISOString(),
    };

    // Add user message immediately
    setMessages((prev) => [...prev, optimisticUserMessage]);

    // Show typing indicator
    setTypingMessage({
      id: 'typing',
      role: 'assistant',
      content: '',
      created_at: new Date().toISOString(),
    });

    setIsSending(true);
    setMessagesError(null);

    try {
      const response = await apiClient.post<SendMessageResponse>(
        `/api/chat/sessions/${sessionId}/messages`,
        { content }
      );

      // Replace optimistic message with real one and add assistant response
      setMessages((prev) => {
        const filtered = prev.filter((m) => m.id !== optimisticUserMessage.id);
        return [...filtered, response.user_message, response.assistant_message];
      });

      // Update session in list with new preview and message count
      setSessions((prev) =>
        prev.map((session) => {
          if (session.id === sessionId) {
            return {
              ...session,
              message_count: session.message_count + 2,
              last_message_preview: content.slice(0, 100),
              updated_at: new Date().toISOString(),
            };
          }
          return session;
        })
      );

      // Move active session to top of list
      setSessions((prev) => {
        const activeSession = prev.find((s) => s.id === sessionId);
        if (activeSession) {
          return [activeSession, ...prev.filter((s) => s.id !== sessionId)];
        }
        return prev;
      });
    } catch (error) {
      console.error('Failed to send message:', error);
      setMessagesError(error instanceof Error ? error.message : 'Failed to send message');

      // Remove optimistic message on error
      setMessages((prev) => prev.filter((m) => m.id !== optimisticUserMessage.id));
    } finally {
      setIsSending(false);
      setTypingMessage(null);
    }
  };

  // Get active session for header
  const activeSession = sessions.find((s) => s.id === activeSessionId);
  const sessionTitle = activeSession?.title || 'New Chat';

  return (
    <div className="flex h-full">
      {/* Sidebar */}
      <SessionSidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        onSelectSession={handleSelectSession}
        onNewSession={handleNewSession}
        isLoading={isLoadingSessions}
      />

      {/* Main chat area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Chat header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900">
          <h1 className="text-lg font-semibold text-gray-900 dark:text-white truncate">
            {sessionTitle}
          </h1>

          {/* Connection status */}
          <div className="flex items-center gap-2">
            {isConnected ? (
              <Badge variant="success">
                <span className="flex items-center gap-1">
                  <Wifi size={12} />
                  Connected
                </span>
              </Badge>
            ) : (
              <Badge variant="danger">
                <span className="flex items-center gap-1">
                  <WifiOff size={12} />
                  Disconnected
                </span>
              </Badge>
            )}
          </div>
        </div>

        {/* Error alerts */}
        {(sessionsError || messagesError) && (
          <div className="px-6 py-2">
            <Alert variant="danger">
              <div className="flex items-center gap-2">
                <AlertCircle size={16} />
                <span>{sessionsError || messagesError}</span>
              </div>
            </Alert>
          </div>
        )}

        {/* Messages */}
        <MessageList
          messages={messages}
          isLoading={isLoadingMessages}
          typingMessage={typingMessage}
        />

        {/* Input */}
        <ChatInput onSend={handleSendMessage} disabled={isSending} />
      </div>
    </div>
  );
}
