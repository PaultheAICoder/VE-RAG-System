import { User, Bot } from 'lucide-react';
import type { ChatMessage } from '../../../types';
import { ConfidenceBadge } from './ConfidenceBadge';
import { CitationCard } from './CitationCard';

interface MessageBubbleProps {
  message: ChatMessage;
  isLoading?: boolean;
}

function formatRelativeTime(dateString: string): string {
  const date = new Date(dateString);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);

  if (diffMins < 1) return 'Just now';
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays < 7) return `${diffDays}d ago`;

  return date.toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  });
}

function TypingIndicator() {
  return (
    <div className="flex gap-1 px-2 py-1">
      <span className="w-2 h-2 bg-gray-400 dark:bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
      <span className="w-2 h-2 bg-gray-400 dark:bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
      <span className="w-2 h-2 bg-gray-400 dark:bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
    </div>
  );
}

export function MessageBubble({ message, isLoading = false }: MessageBubbleProps) {
  const isUser = message.role === 'user';

  return (
    <div className={`flex gap-3 ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
      {/* Avatar */}
      <div
        className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
          isUser
            ? 'bg-primary text-white'
            : 'bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-300'
        }`}
      >
        {isUser ? <User size={18} /> : <Bot size={18} />}
      </div>

      {/* Message content */}
      <div className={`flex flex-col max-w-[75%] ${isUser ? 'items-end' : 'items-start'}`}>
        {/* Role label and time */}
        <div className={`flex items-center gap-2 mb-1 text-xs text-gray-500 dark:text-gray-400 ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
          <span className="font-medium">{isUser ? 'You' : 'Assistant'}</span>
          <span>{formatRelativeTime(message.created_at)}</span>
        </div>

        {/* Message bubble */}
        <div
          className={`rounded-2xl px-4 py-3 ${
            isUser
              ? 'bg-primary text-white rounded-br-md'
              : 'bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-white rounded-bl-md'
          }`}
        >
          {isLoading ? (
            <TypingIndicator />
          ) : (
            <div className="whitespace-pre-wrap break-words">{message.content}</div>
          )}
        </div>

        {/* Assistant-specific elements */}
        {!isUser && !isLoading && (
          <div className="mt-2 space-y-2">
            {/* Citations */}
            {message.sources && message.sources.length > 0 && (
              <CitationCard sources={message.sources} />
            )}

            {/* Confidence badge */}
            {message.confidence && (
              <div className="flex items-center gap-2">
                <ConfidenceBadge confidence={message.confidence} />
              </div>
            )}

            {/* Routing info */}
            {message.was_routed && message.routed_to && (
              <div className="text-xs text-amber-600 dark:text-amber-400 bg-amber-50 dark:bg-amber-900/20 px-2 py-1 rounded">
                Routed to: {message.routed_to}
                {message.route_reason && ` - ${message.route_reason}`}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
