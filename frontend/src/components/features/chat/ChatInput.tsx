import { useState, useRef, useEffect, type KeyboardEvent, type ChangeEvent } from 'react';
import { Send, Command } from 'lucide-react';
import { Button } from '../../ui';

interface ChatInputProps {
  onSend: (message: string) => void;
  disabled?: boolean;
  maxLength?: number;
}

export function ChatInput({ onSend, disabled = false, maxLength = 4000 }: ChatInputProps) {
  const [value, setValue] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
    }
  }, [value]);

  const handleChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    const newValue = e.target.value;
    if (newValue.length <= maxLength) {
      setValue(newValue);
    }
  };

  const handleSend = () => {
    const trimmedValue = value.trim();
    if (trimmedValue && !disabled) {
      onSend(trimmedValue);
      setValue('');
      // Reset textarea height
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    // Cmd/Ctrl + Enter to send
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      e.preventDefault();
      handleSend();
    }
  };

  const canSend = value.trim().length > 0 && !disabled;
  const charCount = value.length;
  const isNearLimit = charCount >= maxLength * 0.9;

  return (
    <div className="border-t border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 p-4">
      <div className="flex gap-3 items-end">
        <div className="flex-1 relative">
          <textarea
            ref={textareaRef}
            value={value}
            onChange={handleChange}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question..."
            disabled={disabled}
            rows={1}
            className={`
              w-full resize-none rounded-xl border px-4 py-3 pr-12
              transition-colors focus:outline-none focus:ring-2
              bg-gray-50 dark:bg-gray-800
              text-gray-900 dark:text-white
              placeholder-gray-400 dark:placeholder-gray-500
              hide-scrollbar
              ${disabled
                ? 'border-gray-200 dark:border-gray-700 cursor-not-allowed opacity-60'
                : 'border-gray-300 dark:border-gray-600 focus:border-primary focus:ring-primary/50'
              }
            `}
          />

          {/* Character count */}
          <div
            className={`absolute right-3 bottom-3 text-xs ${
              isNearLimit
                ? 'text-amber-500 dark:text-amber-400'
                : 'text-gray-400 dark:text-gray-500'
            }`}
          >
            {charCount}/{maxLength}
          </div>
        </div>

        <Button
          onClick={handleSend}
          disabled={!canSend}
          icon={Send}
          className="flex-shrink-0"
        >
          Send
        </Button>
      </div>

      {/* Keyboard shortcut hint */}
      <div className="mt-2 flex items-center gap-1 text-xs text-gray-400 dark:text-gray-500">
        <Command size={12} />
        <span>+ Enter to send</span>
      </div>
    </div>
  );
}
