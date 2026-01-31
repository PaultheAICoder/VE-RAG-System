import type { ReactNode } from 'react';
import { Card } from '../../ui';

interface StatItem {
  label: string;
  value: string | number;
  action?: {
    label: string;
    onClick: () => void;
  };
}

interface StatsCardProps {
  title: string;
  icon: ReactNode;
  stats: StatItem[];
}

export function StatsCard({ title, icon, stats }: StatsCardProps) {
  return (
    <Card variant="elevated">
      <div className="flex items-center gap-3 mb-4">
        <div className="w-10 h-10 rounded-xl flex items-center justify-center bg-primary/10 text-primary">
          {icon}
        </div>
        <h3 className="font-semibold text-gray-900 dark:text-white">{title}</h3>
      </div>
      <dl className="space-y-3">
        {stats.map(({ label, value, action }) => (
          <div key={label} className="flex items-center justify-between">
            <dt className="text-sm text-gray-500 dark:text-gray-400">{label}</dt>
            <dd className="flex items-center gap-2">
              <span className="text-sm font-medium text-gray-900 dark:text-white">{value}</span>
              {action && (
                <button
                  onClick={action.onClick}
                  className="text-xs text-primary hover:underline"
                >
                  [{action.label}]
                </button>
              )}
            </dd>
          </div>
        ))}
      </dl>
    </Card>
  );
}
