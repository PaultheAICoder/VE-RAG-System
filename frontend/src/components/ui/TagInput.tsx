import { useState, useRef, type KeyboardEvent } from 'react';
import { X } from 'lucide-react';

interface TagInputProps {
  label?: string;
  value: string[];
  onChange: (tags: string[]) => void;
  placeholder?: string;
  error?: string;
  disabled?: boolean;
}

export function TagInput({
  label,
  value,
  onChange,
  placeholder = 'Type and press Enter to add',
  error,
  disabled = false,
}: TagInputProps) {
  const [inputValue, setInputValue] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      const trimmed = inputValue.trim().toLowerCase();
      if (trimmed && !value.includes(trimmed)) {
        onChange([...value, trimmed]);
        setInputValue('');
      }
    } else if (e.key === 'Backspace' && !inputValue && value.length > 0) {
      // Remove last tag when backspace on empty input
      onChange(value.slice(0, -1));
    }
  };

  const removeTag = (tagToRemove: string) => {
    if (!disabled) {
      onChange(value.filter((tag) => tag !== tagToRemove));
    }
  };

  const handleContainerClick = () => {
    inputRef.current?.focus();
  };

  return (
    <div>
      {label && (
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5">
          {label}
        </label>
      )}
      <div
        onClick={handleContainerClick}
        className={`
          min-h-[42px] w-full px-3 py-2 rounded-lg border transition-colors cursor-text
          flex flex-wrap gap-2 items-center
          ${error
            ? 'border-red-500 focus-within:ring-red-500/50'
            : 'border-gray-300 dark:border-gray-600 focus-within:border-primary focus-within:ring-primary/50'
          }
          ${disabled ? 'bg-gray-100 dark:bg-gray-900 cursor-not-allowed' : 'bg-white dark:bg-gray-800'}
          focus-within:outline-none focus-within:ring-2
        `}
      >
        {value.map((tag) => (
          <span
            key={tag}
            className={`
              inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-sm
              bg-primary/10 text-primary dark:bg-primary/20 dark:text-primary-light
              ${disabled ? 'opacity-60' : ''}
            `}
          >
            {tag}
            {!disabled && (
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  removeTag(tag);
                }}
                className="hover:bg-primary/20 rounded-full p-0.5 transition-colors"
              >
                <X size={12} />
              </button>
            )}
          </span>
        ))}
        <input
          ref={inputRef}
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={value.length === 0 ? placeholder : ''}
          disabled={disabled}
          className={`
            flex-1 min-w-[120px] outline-none bg-transparent
            text-gray-900 dark:text-white placeholder-gray-400
            ${disabled ? 'cursor-not-allowed' : ''}
          `}
        />
      </div>
      {error && <p className="mt-1 text-sm text-red-500">{error}</p>}
    </div>
  );
}
