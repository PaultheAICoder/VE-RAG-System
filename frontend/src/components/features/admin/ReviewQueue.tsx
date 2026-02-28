import { useState, useEffect, useCallback } from 'react';
import { RefreshCw, CheckCircle, XCircle, Clock, AlertTriangle } from 'lucide-react';
import { Button, Alert, Select, Pagination, Card, Badge } from '../../ui';
import { listReviewQueue, approveReviewItem, rejectReviewItem } from '../../../api/reviewQueue';
import type { ReviewItem, ReviewStatus } from '../../../types';

const STATUS_OPTIONS = [
  { value: 'pending', label: 'Pending' },
  { value: '', label: 'All' },
  { value: 'accepted', label: 'Accepted' },
  { value: 'corrected', label: 'Corrected' },
  { value: 'dismissed', label: 'Dismissed' },
];

const REVIEW_TYPE_LABELS: Record<string, string> = {
  low_confidence_answer: 'Low Confidence Answer',
  account_match_pending: 'Account Match Pending',
  canonicalization_failure: 'Canonicalization Failure',
  unknown_document_type: 'Unknown Document Type',
  ambiguous_classification: 'Ambiguous Classification',
};

const PAGE_SIZE = 20;

function getStatusBadgeVariant(
  reviewStatus: ReviewStatus
): 'default' | 'primary' | 'success' | 'warning' | 'danger' {
  switch (reviewStatus) {
    case 'pending':
      return 'warning';
    case 'accepted':
      return 'success';
    case 'corrected':
      return 'primary';
    case 'dismissed':
      return 'danger';
    default:
      return 'default';
  }
}

function formatConfidence(confidence: number | null): string {
  if (confidence === null || confidence === undefined) return '—';
  return `${(confidence * 100).toFixed(1)}%`;
}

function formatDate(dateStr: string | null): string {
  if (!dateStr) return '—';
  return new Date(dateStr).toLocaleString();
}

interface ReviewItemRowProps {
  item: ReviewItem;
  onApprove: (item: ReviewItem) => void;
  onReject: (item: ReviewItem) => void;
  actionLoading: boolean;
}

function ReviewItemRow({ item, onApprove, onReject, actionLoading }: ReviewItemRowProps) {
  const isPending = item.review_status === 'pending';
  const typeLabel = REVIEW_TYPE_LABELS[item.review_type] ?? item.review_type;

  return (
    <Card className="p-4">
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-2 flex-wrap">
            <Badge variant="default">{typeLabel}</Badge>
            <Badge variant={getStatusBadgeVariant(item.review_status)}>
              {item.review_status.charAt(0).toUpperCase() + item.review_status.slice(1)}
            </Badge>
            {item.confidence !== null && (
              <span className="text-xs text-gray-500 dark:text-gray-400">
                Confidence: {formatConfidence(item.confidence)}
              </span>
            )}
          </div>

          {item.query && (
            <div className="mb-2">
              <span className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide">
                Query
              </span>
              <p className="text-sm text-gray-800 dark:text-gray-200 mt-0.5 line-clamp-2">
                {item.query}
              </p>
            </div>
          )}

          {item.tentative_answer && (
            <div className="mb-2">
              <span className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide">
                Tentative Answer
              </span>
              <p className="text-sm text-gray-700 dark:text-gray-300 mt-0.5 line-clamp-3">
                {item.tentative_answer}
              </p>
            </div>
          )}

          <div className="flex items-center gap-4 text-xs text-gray-400 dark:text-gray-500 mt-2">
            <span className="flex items-center gap-1">
              <Clock size={12} />
              {formatDate(item.created_at)}
            </span>
            {item.resolved_at && (
              <span>Resolved: {formatDate(item.resolved_at)}</span>
            )}
          </div>
        </div>

        {isPending && (
          <div className="flex flex-col gap-2 flex-shrink-0">
            <Button
              variant="ghost"
              size="sm"
              icon={CheckCircle}
              onClick={() => onApprove(item)}
              disabled={actionLoading}
              className="text-green-600 hover:text-green-700 hover:bg-green-50 dark:hover:bg-green-900/20"
              title="Approve"
            >
              Approve
            </Button>
            <Button
              variant="ghost"
              size="sm"
              icon={XCircle}
              onClick={() => onReject(item)}
              disabled={actionLoading}
              className="text-red-500 hover:text-red-600 hover:bg-red-50 dark:hover:bg-red-900/20"
              title="Reject"
            >
              Reject
            </Button>
          </div>
        )}
      </div>
    </Card>
  );
}

export function ReviewQueue() {
  const [items, setItems] = useState<ReviewItem[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [actionLoading, setActionLoading] = useState(false);

  const [statusFilter, setStatusFilter] = useState('pending');
  const [page, setPage] = useState(1);

  const fetchItems = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await listReviewQueue({
        status: statusFilter || undefined,
        page,
        pageSize: PAGE_SIZE,
      });
      setItems(data.items);
      setTotal(data.total);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load review queue');
    } finally {
      setLoading(false);
    }
  }, [statusFilter, page]);

  useEffect(() => {
    fetchItems();
  }, [fetchItems]);

  useEffect(() => {
    if (success) {
      const timer = setTimeout(() => setSuccess(null), 3000);
      return () => clearTimeout(timer);
    }
  }, [success]);

  const handleApprove = async (item: ReviewItem) => {
    setActionLoading(true);
    try {
      await approveReviewItem(item.id);
      setSuccess('Review item approved.');
      await fetchItems();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to approve item');
    } finally {
      setActionLoading(false);
    }
  };

  const handleReject = async (item: ReviewItem) => {
    setActionLoading(true);
    try {
      await rejectReviewItem(item.id);
      setSuccess('Review item rejected.');
      await fetchItems();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to reject item');
    } finally {
      setActionLoading(false);
    }
  };

  const totalPages = Math.ceil(total / PAGE_SIZE);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-heading font-bold text-gray-900 dark:text-white">
            Review Queue
          </h2>
          <p className="text-gray-500 dark:text-gray-400 mt-1">
            Human review for low-confidence answers and classification ambiguities
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            icon={RefreshCw}
            onClick={() => fetchItems()}
            disabled={loading}
            title="Refresh"
          />
        </div>
      </div>

      {error && (
        <Alert variant="danger" onClose={() => setError(null)}>
          {error}
        </Alert>
      )}
      {success && (
        <Alert variant="success" onClose={() => setSuccess(null)}>
          {success}
        </Alert>
      )}

      <div className="flex gap-4 items-center">
        <Select
          options={STATUS_OPTIONS}
          value={statusFilter}
          onChange={(e) => {
            setStatusFilter(e.target.value);
            setPage(1);
          }}
        />
        <span className="text-sm text-gray-500 dark:text-gray-400">
          {total} item{total !== 1 ? 's' : ''}
        </span>
      </div>

      {loading ? (
        <div className="py-12 text-center text-gray-500">Loading review queue...</div>
      ) : items.length === 0 ? (
        <div className="py-12 text-center">
          <AlertTriangle size={32} className="mx-auto mb-3 text-gray-300 dark:text-gray-600" />
          <p className="text-gray-500 dark:text-gray-400">
            {statusFilter === 'pending'
              ? 'No pending items. All reviews are up to date.'
              : 'No items match the selected filter.'}
          </p>
        </div>
      ) : (
        <>
          <div className="space-y-3">
            {items.map((item) => (
              <ReviewItemRow
                key={item.id}
                item={item}
                onApprove={handleApprove}
                onReject={handleReject}
                actionLoading={actionLoading}
              />
            ))}
          </div>

          {totalPages > 1 && (
            <div className="mt-6">
              <Pagination
                currentPage={page}
                totalPages={totalPages}
                totalItems={total}
                itemsPerPage={PAGE_SIZE}
                onPageChange={setPage}
              />
            </div>
          )}
        </>
      )}
    </div>
  );
}
