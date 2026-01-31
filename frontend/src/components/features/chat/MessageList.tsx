import { useEffect, useRef } from 'react';
import { MessageSquare } from 'lucide-react';
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

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, typingMessage]);

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
      className="flex-1 overflow-y-auto px-4 py-6 space-y-6"
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
    </div>
  );
}
