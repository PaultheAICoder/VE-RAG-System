import type { ComponentPropsWithoutRef } from 'react';
import type { LucideIcon } from 'lucide-react';

interface ButtonProps extends ComponentPropsWithoutRef<'button'> {
  variant?: 'primary' | 'secondary' | 'outline' | 'ghost' | 'danger';
  size?: 'sm' | 'md' | 'lg';
  icon?: LucideIcon;
}

export function Button({
  children,
  variant = 'primary',
  size = 'md',
  icon: Icon,
  disabled = false,
  className = '',
  ...props
}: ButtonProps) {
  const base = 'inline-flex items-center justify-center font-semibold transition-all duration-200 rounded-lg gap-2 disabled:opacity-50 disabled:cursor-not-allowed';

  const variants = {
    primary: 'bg-primary hover:bg-primary-dark text-white shadow-md hover:shadow-lg',
    secondary: 'bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-900 dark:text-white',
    outline: 'border-2 border-primary text-primary hover:bg-primary hover:text-white',
    ghost: 'text-primary hover:bg-primary/10',
    danger: 'bg-red-500 hover:bg-red-600 text-white',
  };

  const sizes = {
    sm: 'px-3 py-1.5 text-sm',
    md: 'px-5 py-2.5 text-base',
    lg: 'px-7 py-3.5 text-lg',
  };

  return (
    <button
      className={`${base} ${variants[variant]} ${sizes[size]} ${className}`}
      disabled={disabled}
      {...props}
    >
      {Icon && <Icon size={size === 'sm' ? 16 : size === 'lg' ? 22 : 18} />}
      {children}
    </button>
  );
}
