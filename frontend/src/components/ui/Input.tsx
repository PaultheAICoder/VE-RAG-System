import type { ComponentPropsWithoutRef } from 'react';
import type { LucideIcon } from 'lucide-react';

interface InputProps extends ComponentPropsWithoutRef<'input'> {
  label?: string;
  error?: string;
  icon?: LucideIcon;
}

export function Input({
  label,
  error,
  icon: Icon,
  className = '',
  ...props
}: InputProps) {
  return (
    <div className={className}>
      {label && (
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5">
          {label}
        </label>
      )}
      <div className="relative">
        {Icon && (
          <Icon
            className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400"
            size={18}
          />
        )}
        <input
          className={`
            w-full px-4 py-2.5 rounded-lg border transition-colors
            ${Icon ? 'pl-10' : ''}
            ${error
              ? 'border-red-500 focus:ring-red-500/50'
              : 'border-gray-300 dark:border-gray-600 focus:border-primary focus:ring-primary/50'
            }
            bg-white dark:bg-gray-800
            text-gray-900 dark:text-white
            placeholder-gray-400
            focus:outline-none focus:ring-2
          `}
          {...props}
        />
      </div>
      {error && <p className="mt-1 text-sm text-red-500">{error}</p>}
    </div>
  );
}
