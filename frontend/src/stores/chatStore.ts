import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import type { ChatSession, ChatMessage } from '../types';

// Track hydration state for components that need to wait
let hasHydrated = false;
const hydrationListeners: (() => void)[] = [];

export const waitForHydration = (): Promise<void> => {
  if (hasHydrated) return Promise.resolve();
  return new Promise((resolve) => {
    hydrationListeners.push(resolve);
  });
};

export const useHasHydrated = () => {
  const [hydrated, setHydrated] = React.useState(hasHydrated);
  React.useEffect(() => {
    if (hasHydrated) {
      setHydrated(true);
      return;
    }
    const listener = () => setHydrated(true);
    hydrationListeners.push(listener);
    return () => {
      const idx = hydrationListeners.indexOf(listener);
      if (idx !== -1) hydrationListeners.splice(idx, 1);
    };
  }, []);
  return hydrated;
};

// Need React for the hook
import * as React from 'react';

interface ChatState {
  // Session state
  sessions: ChatSession[];
  activeSessionId: string | null;

  // Messages for active session (cached)
  messages: ChatMessage[];

  // Loading/error state (ephemeral - not persisted)
  isLoadingSessions: boolean;
  isLoadingMessages: boolean;
  isSending: boolean;
  sessionsError: string | null;
  messagesError: string | null;

  // Pending response tracking
  pendingMessageId: string | null;
  typingMessage: ChatMessage | null;

  // Session actions
  setSessions: (sessions: ChatSession[]) => void;
  setActiveSession: (sessionId: string | null) => void;
  addSession: (session: ChatSession) => void;
  updateSession: (id: string, updates: Partial<ChatSession>) => void;
  removeSession: (sessionId: string) => void;
  removeSessions: (sessionIds: string[]) => void;
  moveSessionToTop: (sessionId: string) => void;

  // Message actions
  setMessages: (messages: ChatMessage[]) => void;
  addMessage: (message: ChatMessage) => void;
  updateMessage: (id: string, updates: Partial<ChatMessage>) => void;
  replaceOptimisticMessage: (tempId: string, realMessage: ChatMessage) => void;
  removeMessage: (id: string) => void;

  // Loading state actions
  setLoadingSessions: (loading: boolean) => void;
  setLoadingMessages: (loading: boolean) => void;
  setSending: (sending: boolean) => void;
  setTypingMessage: (message: ChatMessage | null) => void;

  // Error actions
  setSessionsError: (error: string | null) => void;
  setMessagesError: (error: string | null) => void;

  // Pending state
  setPendingMessageId: (id: string | null) => void;

  // Sync and reset
  syncFromBackend: (messages: ChatMessage[]) => void;
  clearSession: () => void;
  reset: () => void;
}

const initialState = {
  sessions: [] as ChatSession[],
  activeSessionId: null as string | null,
  messages: [] as ChatMessage[],
  isLoadingSessions: false,
  isLoadingMessages: false,
  isSending: false,
  sessionsError: null as string | null,
  messagesError: null as string | null,
  pendingMessageId: null as string | null,
  typingMessage: null as ChatMessage | null,
};

export const useChatStore = create<ChatState>()(
  persist(
    (set, get) => ({
      ...initialState,

      // Session actions
      setSessions: (sessions) => set({ sessions }),

      setActiveSession: (sessionId) => set({ activeSessionId: sessionId }),

      addSession: (session) =>
        set((state) => ({
          sessions: [session, ...state.sessions],
        })),

      updateSession: (id, updates) =>
        set((state) => ({
          sessions: state.sessions.map((s) =>
            s.id === id ? { ...s, ...updates } : s
          ),
        })),

      removeSession: (sessionId) =>
        set((state) => {
          const newSessions = state.sessions.filter((s) => s.id !== sessionId);
          // If the deleted session was active, clear the active session
          const newActiveId =
            state.activeSessionId === sessionId
              ? newSessions.length > 0
                ? newSessions[0].id
                : null
              : state.activeSessionId;
          return {
            sessions: newSessions,
            activeSessionId: newActiveId,
            // Clear messages if the active session was deleted
            ...(state.activeSessionId === sessionId
              ? { messages: [], messagesError: null }
              : {}),
          };
        }),

      removeSessions: (sessionIds) =>
        set((state) => {
          const idsSet = new Set(sessionIds);
          const newSessions = state.sessions.filter((s) => !idsSet.has(s.id));
          // If the active session was deleted, select the first remaining session
          const activeDeleted = state.activeSessionId && idsSet.has(state.activeSessionId);
          const newActiveId = activeDeleted
            ? newSessions.length > 0
              ? newSessions[0].id
              : null
            : state.activeSessionId;
          return {
            sessions: newSessions,
            activeSessionId: newActiveId,
            // Clear messages if the active session was deleted
            ...(activeDeleted ? { messages: [], messagesError: null } : {}),
          };
        }),

      moveSessionToTop: (sessionId) =>
        set((state) => {
          const session = state.sessions.find((s) => s.id === sessionId);
          if (!session) return state;
          return {
            sessions: [session, ...state.sessions.filter((s) => s.id !== sessionId)],
          };
        }),

      // Message actions
      setMessages: (messages) => {
        console.log('[ChatStore] setMessages', { count: messages.length });
        set({ messages });
      },

      addMessage: (message) => {
        console.log('[ChatStore] addMessage', {
          id: message.id,
          role: message.role,
          contentPreview: message.content?.slice(0, 50),
        });
        set((state) => ({
          messages: [...state.messages, message],
        }));
      },

      updateMessage: (id, updates) =>
        set((state) => ({
          messages: state.messages.map((m) =>
            m.id === id ? { ...m, ...updates } : m
          ),
        })),

      replaceOptimisticMessage: (tempId, realMessage) => {
        const { messages } = get();
        const found = messages.some((m) => m.id === tempId);
        console.log('[ChatStore] replaceOptimisticMessage', {
          tempId,
          realId: realMessage.id,
          found,
          messageIds: messages.map((m) => m.id),
        });
        set((state) => ({
          messages: state.messages.map((m) =>
            m.id === tempId ? realMessage : m
          ),
        }));
      },

      removeMessage: (id) =>
        set((state) => ({
          messages: state.messages.filter((m) => m.id !== id),
        })),

      // Loading state actions
      setLoadingSessions: (loading) => set({ isLoadingSessions: loading }),
      setLoadingMessages: (loading) => set({ isLoadingMessages: loading }),
      setSending: (sending) => set({ isSending: sending }),
      setTypingMessage: (message) => set({ typingMessage: message }),

      // Error actions
      setSessionsError: (error) => set({ sessionsError: error }),
      setMessagesError: (error) => set({ messagesError: error }),

      // Pending state
      setPendingMessageId: (id) => set({ pendingMessageId: id }),

      // Sync from backend (backend is source of truth)
      syncFromBackend: (backendMessages) => {
        const { pendingMessageId, messages: currentMessages } = get();

        console.log('[ChatStore] syncFromBackend called', {
          pendingMessageId,
          currentMessageCount: currentMessages.length,
          backendMessageCount: backendMessages.length,
          currentIds: currentMessages.map((m) => m.id),
          backendIds: backendMessages.map((m) => m.id),
        });

        // If we have a pending message (send in progress), merge carefully
        if (pendingMessageId) {
          const pendingMessage = currentMessages.find((m) => m.id === pendingMessageId);

          if (pendingMessage) {
            console.log('[ChatStore] Has pending message, preserving it', {
              pendingId: pendingMessageId,
              pendingContent: pendingMessage.content?.slice(0, 50),
            });

            // Check if backend already has this message (by content match since IDs differ)
            const backendHasPending = backendMessages.some(
              (m) => m.role === 'user' && m.content === pendingMessage.content
            );

            if (!backendHasPending) {
              // Backend doesn't have our pending message yet, preserve it
              console.log('[ChatStore] Backend missing pending, preserving optimistic message');
              set({ messages: [...backendMessages, pendingMessage] });
              return;
            }
          }
        }

        // Check if response has arrived (new assistant message)
        let shouldClearPending = false;
        if (pendingMessageId) {
          const lastAssistantMessage = [...backendMessages]
            .reverse()
            .find((m) => m.role === 'assistant');

          const currentLastAssistant = [...currentMessages]
            .reverse()
            .find((m) => m.role === 'assistant');

          if (
            lastAssistantMessage &&
            (!currentLastAssistant ||
              lastAssistantMessage.id !== currentLastAssistant.id)
          ) {
            shouldClearPending = true;
            console.log('[ChatStore] New assistant message detected, clearing pending');
          }
        }

        set({
          messages: backendMessages,
          ...(shouldClearPending
            ? { pendingMessageId: null, typingMessage: null }
            : {}),
        });
      },

      // Clear current session state (e.g., when creating new session)
      clearSession: () =>
        set({
          activeSessionId: null,
          messages: [],
          pendingMessageId: null,
          typingMessage: null,
          messagesError: null,
        }),

      // Full reset
      reset: () => set(initialState),
    }),
    {
      name: 'chat-storage',
      storage: createJSONStorage(() => sessionStorage),
      // Only persist these fields (loading/error states are ephemeral)
      partialize: (state) => ({
        activeSessionId: state.activeSessionId,
        sessions: state.sessions,
        messages: state.messages,
        pendingMessageId: state.pendingMessageId,
      }),
      onRehydrateStorage: () => (_state) => {
        // Mark hydration complete and notify listeners
        hasHydrated = true;
        hydrationListeners.forEach((listener) => listener());
        hydrationListeners.length = 0;
      },
    }
  )
);
