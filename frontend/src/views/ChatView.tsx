import { useEffect, useCallback } from 'react';
import { Wifi, WifiOff, AlertCircle } from 'lucide-react';
import { apiClient } from '../api/client';
import type {
  ChatSession,
  ChatMessage,
  SessionListResponse,
  MessageListResponse,
  SendMessageResponse,
} from '../types';
import { useChatStore, useHasHydrated } from '../stores/chatStore';
import { useAuthStore } from '../stores/authStore';
import { SessionSidebar } from '../components/features/chat/SessionSidebar';
import { MessageList } from '../components/features/chat/MessageList';
import { ChatInput } from '../components/features/chat/ChatInput';
import { Alert, Badge } from '../components/ui';

export function ChatView() {
  // Get state and actions from Zustand store
  const {
    sessions,
    activeSessionId,
    messages,
    isLoadingSessions,
    isLoadingMessages,
    isSending,
    sessionsError,
    messagesError,
    typingMessage,
    pendingMessageId,
    setSessions,
    setActiveSession,
    addSession,
    updateSession,
    moveSessionToTop,
    setMessages,
    addMessage,
    replaceOptimisticMessage,
    removeMessage,
    setLoadingSessions,
    setLoadingMessages,
    setSending,
    setTypingMessage,
    setSessionsError,
    setMessagesError,
    setPendingMessageId,
    syncFromBackend,
  } = useChatStore();

  // Connection status (for future WebSocket, currently shows HTTP)
  const isConnected = true;

  // Check if user is admin (for session deletion features)
  const user = useAuthStore((state) => state.user);
  const isAdmin = user?.role === 'admin' || user?.role === 'customer_admin';

  // Wait for store hydration before loading
  const hasHydrated = useHasHydrated();

  // Load sessions on mount and sync with backend (only after hydration)
  useEffect(() => {
    if (hasHydrated) {
      loadSessions();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hasHydrated]);

  // Load messages when active session changes
  useEffect(() => {
    if (activeSessionId) {
      loadMessages(activeSessionId);
    } else {
      setMessages([]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeSessionId]);

  // Handle pending message recovery on mount
  useEffect(() => {
    // If there's a pending message ID, the user may have navigated away during a request
    // When they return, we should check if the response has arrived
    if (pendingMessageId && activeSessionId) {
      // The loadMessages call will trigger syncFromBackend which handles this
      loadMessages(activeSessionId);
    }
    // Only run on mount
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const loadSessions = async () => {
    setLoadingSessions(true);
    setSessionsError(null);
    try {
      const response = await apiClient.get<SessionListResponse>('/api/chat/sessions?limit=50');
      setSessions(response.sessions);

      // Auto-select first session if none selected and sessions exist
      // Use store's activeSessionId which persists across navigation
      const currentActiveId = useChatStore.getState().activeSessionId;
      if (response.sessions.length > 0 && !currentActiveId) {
        setActiveSession(response.sessions[0].id);
      } else if (currentActiveId) {
        // Verify the active session still exists
        const sessionExists = response.sessions.some((s) => s.id === currentActiveId);
        if (!sessionExists) {
          // Session was deleted while user was away
          setActiveSession(response.sessions.length > 0 ? response.sessions[0].id : null);
        }
      }
    } catch (error) {
      console.error('Failed to load sessions:', error);
      setSessionsError(error instanceof Error ? error.message : 'Failed to load sessions');
    } finally {
      setLoadingSessions(false);
    }
  };

  const loadMessages = async (sessionId: string) => {
    setLoadingMessages(true);
    setMessagesError(null);
    try {
      const response = await apiClient.get<MessageListResponse>(
        `/api/chat/sessions/${sessionId}/messages?limit=50`
      );
      // Use syncFromBackend to handle pending message recovery
      syncFromBackend(response.messages);
    } catch (error) {
      console.error('Failed to load messages:', error);
      setMessagesError(error instanceof Error ? error.message : 'Failed to load messages');
    } finally {
      setLoadingMessages(false);
    }
  };

  const createSession = async (): Promise<string | null> => {
    try {
      const response = await apiClient.post<ChatSession>('/api/chat/sessions', {
        title: null,
      });

      // Add new session to the top of the list
      addSession({
        ...response,
        message_count: 0,
        last_message_preview: undefined,
      });

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
      setActiveSession(sessionId);
      setMessages([]);
      setPendingMessageId(null);
      setTypingMessage(null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
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

  const handleSelectSession = useCallback(
    (sessionId: string) => {
      setActiveSession(sessionId);
    },
    [setActiveSession]
  );

  const handleSendMessage = async (content: string) => {
    if (!content.trim() || isSending) return;

    let sessionId = activeSessionId;

    // Create a new session if none exists
    if (!sessionId) {
      sessionId = await createSession();
      if (!sessionId) return;
      setActiveSession(sessionId);
    }

    // Create optimistic user message
    const tempId = `temp-${Date.now()}`;
    const optimisticUserMessage: ChatMessage = {
      id: tempId,
      role: 'user',
      content,
      created_at: new Date().toISOString(),
    };

    // Add user message immediately (optimistic update)
    addMessage(optimisticUserMessage);

    // Track pending message for recovery if user navigates away
    setPendingMessageId(tempId);

    // Show typing indicator
    setTypingMessage({
      id: 'typing',
      role: 'assistant',
      content: '',
      created_at: new Date().toISOString(),
    });

    setSending(true);
    setMessagesError(null);

    try {
      const response = await apiClient.post<SendMessageResponse>(
        `/api/chat/sessions/${sessionId}/messages`,
        { content }
      );

      // Replace optimistic message with real one
      replaceOptimisticMessage(tempId, response.user_message);

      // Add assistant response
      addMessage(response.assistant_message);

      // Clear pending state
      setPendingMessageId(null);

      // Update session in list with new preview and message count
      updateSession(sessionId, {
        message_count:
          (sessions.find((s) => s.id === sessionId)?.message_count ?? 0) + 2,
        last_message_preview: content.slice(0, 100),
        updated_at: new Date().toISOString(),
      });

      // Move active session to top of list
      moveSessionToTop(sessionId);
    } catch (error) {
      console.error('Failed to send message:', error);
      setMessagesError(error instanceof Error ? error.message : 'Failed to send message');

      // Remove optimistic message on error
      removeMessage(tempId);

      // Clear pending state
      setPendingMessageId(null);
    } finally {
      setSending(false);
      setTypingMessage(null);
    }
  };

  // Get active session for header
  const activeSession = sessions.find((s) => s.id === activeSessionId);
  const sessionTitle = activeSession?.title || 'New Chat';

  return (
    <div className="flex h-[calc(100vh-140px)]">
      {/* Sidebar */}
      <SessionSidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        onSelectSession={handleSelectSession}
        onNewSession={handleNewSession}
        isLoading={isLoadingSessions}
        isAdmin={isAdmin}
      />

      {/* Main chat area */}
      <div className="flex-1 flex flex-col min-w-0 min-h-0">
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
