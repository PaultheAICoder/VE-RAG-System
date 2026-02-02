import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import type { ChatSession, ChatMessage } from '../types';

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

      moveSessionToTop: (sessionId) =>
        set((state) => {
          const session = state.sessions.find((s) => s.id === sessionId);
          if (!session) return state;
          return {
            sessions: [session, ...state.sessions.filter((s) => s.id !== sessionId)],
          };
        }),

      // Message actions
      setMessages: (messages) => set({ messages }),

      addMessage: (message) =>
        set((state) => ({
          messages: [...state.messages, message],
        })),

      updateMessage: (id, updates) =>
        set((state) => ({
          messages: state.messages.map((m) =>
            m.id === id ? { ...m, ...updates } : m
          ),
        })),

      replaceOptimisticMessage: (tempId, realMessage) =>
        set((state) => ({
          messages: state.messages.map((m) =>
            m.id === tempId ? realMessage : m
          ),
        })),

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
      syncFromBackend: (messages) => {
        const { pendingMessageId } = get();

        // If we have a pending message ID, check if response has arrived
        // A response is indicated by an assistant message following the user message
        let shouldClearPending = false;
        if (pendingMessageId) {
          // Find index of the pending user message (the temp ID might have been replaced)
          // Check if there's an assistant message that wasn't there before
          const lastAssistantMessage = [...messages]
            .reverse()
            .find((m) => m.role === 'assistant');

          // If the last assistant message is newer than what we had, response arrived
          const currentMessages = get().messages;
          const currentLastAssistant = [...currentMessages]
            .reverse()
            .find((m) => m.role === 'assistant');

          if (
            lastAssistantMessage &&
            (!currentLastAssistant ||
              lastAssistantMessage.id !== currentLastAssistant.id)
          ) {
            shouldClearPending = true;
          }
        }

        set({
          messages,
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
    }
  )
);
