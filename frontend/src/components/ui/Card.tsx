import type { ReactNode } from 'react';

interface CardProps {
  children: ReactNode;
  variant?: 'default' | 'elevated' | 'primary' | 'cream';
  className?: string;
}

export function Card({
  children,
  variant = 'default',
  className = '',
}: CardProps) {
  const variants = {
    default: 'bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700',
    elevated: 'bg-white dark:bg-gray-800 shadow-lg border border-gray-100 dark:border-gray-700',
    primary: 'bg-primary text-white',
    cream: 'bg-cream dark:bg-gray-800 border border-gray-200 dark:border-gray-700',
  };

  return (
    <div className={`rounded-xl p-6 ${variants[variant]} ${className}`}>
      {children}
    </div>
  );
}
