import { useEffect, useRef, useState, useCallback } from 'react';
import { MessageSquare, ArrowDown } from 'lucide-react';
import type { ChatMessage } from '../../../types';
import { MessageBubble } from './MessageBubble';

interface MessageListProps {
  messages: ChatMessage[];
  isLoading?: boolean;
  typingMessage?: ChatMessage | null;
}

export function MessageList({ messages, isLoading = false, typingMessage }: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [isAtBottom, setIsAtBottom] = useState(true);

  const checkIfAtBottom = useCallback(() => {
    const container = containerRef.current;
    if (!container) return true;
    const threshold = 100; // pixels from bottom
    return container.scrollHeight - container.scrollTop - container.clientHeight < threshold;
  }, []);

  const handleScroll = useCallback(() => {
    setIsAtBottom(checkIfAtBottom());
  }, [checkIfAtBottom]);

  const scrollToBottom = useCallback(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  // Auto-scroll to bottom when new messages arrive (only if already at bottom)
  useEffect(() => {
    if (isAtBottom && bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, typingMessage, isAtBottom]);

  // Empty state
  if (messages.length === 0 && !typingMessage && !isLoading) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center text-gray-500 dark:text-gray-400">
        <MessageSquare size={48} className="mb-4 opacity-50" />
        <h3 className="text-lg font-medium mb-1">Start a conversation</h3>
        <p className="text-sm">Ask a question about your documents</p>
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      onScroll={handleScroll}
      className="flex-1 min-h-0 relative overflow-y-auto px-4 py-6 space-y-6"
    >
      {/* Loading skeleton for initial load */}
      {isLoading && messages.length === 0 && (
        <div className="space-y-6">
          {[1, 2, 3].map((i) => (
            <div key={i} className="flex gap-3 animate-pulse">
              <div className="w-8 h-8 bg-gray-200 dark:bg-gray-700 rounded-full" />
              <div className="space-y-2">
                <div className="h-4 w-24 bg-gray-200 dark:bg-gray-700 rounded" />
                <div className="h-16 w-64 bg-gray-200 dark:bg-gray-700 rounded-2xl" />
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Messages */}
      {messages.map((message) => (
        <MessageBubble key={message.id} message={message} />
      ))}

      {/* Typing indicator for pending response */}
      {typingMessage && (
        <MessageBubble message={typingMessage} isLoading />
      )}

      {/* Scroll anchor */}
      <div ref={bottomRef} />

      {/* Scroll to bottom button */}
      {!isAtBottom && (
        <button
          onClick={scrollToBottom}
          className="absolute bottom-4 right-4 p-2 rounded-full bg-primary text-white shadow-lg hover:bg-primary-dark transition-all opacity-90 hover:opacity-100"
          aria-label="Scroll to bottom"
        >
          <ArrowDown size={20} />
        </button>
      )}
    </div>
  );
}
