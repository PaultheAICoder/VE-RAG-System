import { CheckCircle, Clock, Loader2, XCircle } from 'lucide-react';
import { Badge } from '../../ui';
import type { DocumentStatus } from '../../../types';

interface StatusBadgeProps {
  status: DocumentStatus;
}

const statusConfig = {
  ready: {
    variant: 'success' as const,
    icon: CheckCircle,
    label: 'Ready',
  },
  pending: {
    variant: 'default' as const,
    icon: Clock,
    label: 'Queued',
  },
  processing: {
    variant: 'warning' as const,
    icon: Loader2,
    label: 'Processing',
  },
  failed: {
    variant: 'danger' as const,
    icon: XCircle,
    label: 'Failed',
  },
};

export function StatusBadge({ status }: StatusBadgeProps) {
  const config = statusConfig[status];
  const Icon = config.icon;

  return (
    <Badge variant={config.variant}>
      <span className="inline-flex items-center gap-1">
        <Icon
          size={12}
          className={
            status === 'processing' ? 'animate-spin' :
            status === 'pending' ? 'animate-pulse' : ''
          }
        />
        {config.label}
      </span>
    </Badge>
  );
}
