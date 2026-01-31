import type { ReactNode } from 'react';
import { Info, CheckCircle, AlertTriangle, XCircle, X } from 'lucide-react';

interface AlertProps {
  children: ReactNode;
  variant?: 'info' | 'success' | 'warning' | 'danger';
  title?: string;
  onClose?: () => void;
}

export function Alert({
  children,
  variant = 'info',
  title,
  onClose,
}: AlertProps) {
  const config = {
    info: {
      styles: 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800 text-blue-800 dark:text-blue-300',
      icon: Info,
    },
    success: {
      styles: 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800 text-green-800 dark:text-green-300',
      icon: CheckCircle,
    },
    warning: {
      styles: 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800 text-yellow-800 dark:text-yellow-300',
      icon: AlertTriangle,
    },
    danger: {
      styles: 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800 text-red-800 dark:text-red-300',
      icon: XCircle,
    },
  };

  const { styles, icon: Icon } = config[variant];

  return (
    <div className={`flex items-start gap-3 p-4 rounded-lg border ${styles}`}>
      <Icon size={20} className="flex-shrink-0 mt-0.5" />
      <div className="flex-1">
        {title && <div className="font-semibold mb-1">{title}</div>}
        <div>{children}</div>
      </div>
      {onClose && (
        <button onClick={onClose} className="opacity-70 hover:opacity-100">
          <X size={18} />
        </button>
      )}
    </div>
  );
}
