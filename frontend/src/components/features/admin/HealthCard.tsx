import type { ReactNode } from 'react';
import { Card } from '../../ui';

interface HealthCardProps {
  title: string;
  status: 'healthy' | 'unhealthy' | 'unknown';
  icon: ReactNode;
  details: { label: string; value: string }[];
}

export function HealthCard({ title, status, icon, details }: HealthCardProps) {
  const statusColors = {
    healthy: 'text-green-500',
    unhealthy: 'text-red-500',
    unknown: 'text-gray-400',
  };

  const statusIndicator = {
    healthy: 'bg-green-500',
    unhealthy: 'bg-red-500',
    unknown: 'bg-gray-400',
  };

  return (
    <Card variant="elevated" className="relative overflow-hidden">
      <div className="flex items-start gap-4">
        <div
          className={`flex-shrink-0 w-12 h-12 rounded-xl flex items-center justify-center bg-gray-100 dark:bg-gray-800 ${statusColors[status]}`}
        >
          {icon}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-2">
            <span
              className={`w-2.5 h-2.5 rounded-full ${statusIndicator[status]}`}
              title={status}
            />
            <h3 className="font-semibold text-gray-900 dark:text-white">{title}</h3>
          </div>
          <dl className="space-y-1">
            {details.map(({ label, value }) => (
              <div key={label} className="flex items-center gap-2 text-sm">
                <dt className="text-gray-500 dark:text-gray-400">{label}:</dt>
                <dd className="text-gray-700 dark:text-gray-300 truncate">{value}</dd>
              </div>
            ))}
          </dl>
        </div>
      </div>
    </Card>
  );
}
