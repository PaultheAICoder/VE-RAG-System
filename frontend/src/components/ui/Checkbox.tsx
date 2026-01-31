import type { ComponentPropsWithoutRef } from 'react';
import { Check, Minus } from 'lucide-react';

interface CheckboxProps extends Omit<ComponentPropsWithoutRef<'input'>, 'type'> {
  label?: string;
  indeterminate?: boolean;
}

export function Checkbox({
  label,
  indeterminate = false,
  checked = false,
  className = '',
  ...props
}: CheckboxProps) {
  const isChecked = checked || indeterminate;

  return (
    <label className={`inline-flex items-center gap-2 cursor-pointer ${className}`}>
      <div className="relative">
        <input
          type="checkbox"
          checked={checked}
          className="sr-only peer"
          {...props}
        />
        <div
          className={`
            w-5 h-5 rounded border-2 transition-all
            flex items-center justify-center
            ${isChecked
              ? 'bg-primary border-primary'
              : 'bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-600'
            }
            peer-focus:ring-2 peer-focus:ring-primary/50 peer-focus:ring-offset-2
            dark:peer-focus:ring-offset-gray-900
          `}
        >
          {indeterminate ? (
            <Minus size={14} className="text-white" />
          ) : checked ? (
            <Check size={14} className="text-white" />
          ) : null}
        </div>
      </div>
      {label && (
        <span className="text-sm text-gray-700 dark:text-gray-300">{label}</span>
      )}
    </label>
  );
}
