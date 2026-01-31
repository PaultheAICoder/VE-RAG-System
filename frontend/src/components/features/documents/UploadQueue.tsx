import { FileText, Check, X, Loader2, Clock } from 'lucide-react';
import { Button } from '../../ui';

export type UploadStatus = 'queued' | 'uploading' | 'done' | 'failed';

export interface QueuedFile {
  id: string;
  file: File;
  status: UploadStatus;
  progress: number;
  error?: string;
}

interface UploadQueueProps {
  files: QueuedFile[];
  onRemove: (id: string) => void;
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function getStatusIcon(status: UploadStatus) {
  switch (status) {
    case 'queued':
      return <Clock size={16} className="text-gray-400" />;
    case 'uploading':
      return <Loader2 size={16} className="text-primary animate-spin" />;
    case 'done':
      return <Check size={16} className="text-green-500" />;
    case 'failed':
      return <X size={16} className="text-red-500" />;
  }
}

export function UploadQueue({ files, onRemove }: UploadQueueProps) {
  if (files.length === 0) {
    return null;
  }

  return (
    <div className="space-y-2">
      <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300">
        Queued Files ({files.length})
      </h4>

      <div className="space-y-2 max-h-48 overflow-y-auto">
        {files.map((qf) => (
          <div
            key={qf.id}
            className={`
              flex items-center gap-3 p-3 rounded-lg border
              ${qf.status === 'failed'
                ? 'border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-900/20'
                : 'border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50'
              }
            `}
          >
            <FileText size={20} className="text-gray-400 flex-shrink-0" />

            <div className="flex-1 min-w-0">
              <div className="flex items-center justify-between gap-2">
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300 truncate">
                  {qf.file.name}
                </span>
                <span className="text-xs text-gray-500 flex-shrink-0">
                  {formatFileSize(qf.file.size)}
                </span>
              </div>

              {/* Progress bar */}
              {qf.status === 'uploading' && (
                <div className="mt-2">
                  <div className="h-1.5 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-primary transition-all duration-300"
                      style={{ width: `${qf.progress}%` }}
                    />
                  </div>
                  <span className="text-xs text-gray-500 mt-1">{qf.progress}%</span>
                </div>
              )}

              {/* Error message */}
              {qf.status === 'failed' && qf.error && (
                <p className="text-xs text-red-600 dark:text-red-400 mt-1 truncate">
                  {qf.error}
                </p>
              )}
            </div>

            <div className="flex items-center gap-2 flex-shrink-0">
              {getStatusIcon(qf.status)}

              {(qf.status === 'queued' || qf.status === 'failed') && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => onRemove(qf.id)}
                  className="!p-1"
                >
                  <X size={14} />
                </Button>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
