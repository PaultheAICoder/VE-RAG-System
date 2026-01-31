import type { ComponentPropsWithoutRef } from 'react';

interface SliderProps extends Omit<ComponentPropsWithoutRef<'input'>, 'type'> {
  label?: string;
  description?: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  showValue?: boolean;
  valueFormatter?: (value: number) => string;
}

export function Slider({
  label,
  description,
  value,
  min,
  max,
  step = 1,
  showValue = true,
  valueFormatter,
  className = '',
  ...props
}: SliderProps) {
  const displayValue = valueFormatter ? valueFormatter(value) : value.toString();

  // Calculate percentage for track fill
  const percentage = ((value - min) / (max - min)) * 100;

  return (
    <div className={className}>
      {(label || showValue) && (
        <div className="flex items-center justify-between mb-2">
          {label && (
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
              {label}
            </label>
          )}
          {showValue && (
            <span className="text-sm font-medium text-gray-900 dark:text-white tabular-nums">
              {displayValue}
            </span>
          )}
        </div>
      )}
      {description && (
        <p className="text-xs text-gray-500 dark:text-gray-400 mb-2">{description}</p>
      )}
      <div className="relative">
        <input
          type="range"
          value={value}
          min={min}
          max={max}
          step={step}
          className="
            w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer
            [&::-webkit-slider-thumb]:appearance-none
            [&::-webkit-slider-thumb]:w-4
            [&::-webkit-slider-thumb]:h-4
            [&::-webkit-slider-thumb]:rounded-full
            [&::-webkit-slider-thumb]:bg-primary
            [&::-webkit-slider-thumb]:shadow-md
            [&::-webkit-slider-thumb]:cursor-pointer
            [&::-webkit-slider-thumb]:transition-transform
            [&::-webkit-slider-thumb]:hover:scale-110
            [&::-moz-range-thumb]:w-4
            [&::-moz-range-thumb]:h-4
            [&::-moz-range-thumb]:rounded-full
            [&::-moz-range-thumb]:bg-primary
            [&::-moz-range-thumb]:border-0
            [&::-moz-range-thumb]:shadow-md
            [&::-moz-range-thumb]:cursor-pointer
            focus:outline-none focus:ring-2 focus:ring-primary/50 focus:ring-offset-2
            dark:focus:ring-offset-gray-800
          "
          style={{
            background: `linear-gradient(to right, var(--color-primary, #3b82f6) 0%, var(--color-primary, #3b82f6) ${percentage}%, rgb(229 231 235) ${percentage}%, rgb(229 231 235) 100%)`,
          }}
          {...props}
        />
      </div>
      <div className="flex justify-between mt-1 text-xs text-gray-400 dark:text-gray-500">
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  );
}
