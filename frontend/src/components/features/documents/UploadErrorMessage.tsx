import { AlertTriangle, Copy } from 'lucide-react';
import type { UploadErrorResponse } from '../../../types';

interface UploadErrorMessageProps {
  error: string;
  duplicateInfo?: UploadErrorResponse;
}

/**
 * Display upload error with detailed info for duplicates.
 */
export function UploadErrorMessage({ error, duplicateInfo }: UploadErrorMessageProps) {
  const isDuplicate = duplicateInfo?.error_code === 'DUPLICATE_FILE';

  if (isDuplicate && duplicateInfo) {
    return (
      <div className="flex items-start gap-2 text-xs text-red-600 dark:text-red-400 mt-1">
        <Copy size={14} className="flex-shrink-0 mt-0.5" />
        <div>
          <span className="font-medium">Duplicate file</span>
          <span className="text-gray-500 dark:text-gray-400">
            {' '}
            - matches &quot;{duplicateInfo.existing_filename}&quot;
          </span>
          {duplicateInfo.uploaded_at && (
            <span className="text-gray-400 dark:text-gray-500 text-[10px] block">
              Uploaded {new Date(duplicateInfo.uploaded_at).toLocaleDateString()}
            </span>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="flex items-start gap-2 text-xs text-red-600 dark:text-red-400 mt-1">
      <AlertTriangle size={14} className="flex-shrink-0 mt-0.5" />
      <span className="truncate">{error}</span>
    </div>
  );
}

/**
 * Map error code to user-friendly message.
 */
export function getErrorMessage(errorResponse?: UploadErrorResponse): string {
  if (!errorResponse) return 'Upload failed';

  switch (errorResponse.error_code) {
    case 'DUPLICATE_FILE':
      return `Duplicate of "${errorResponse.existing_filename}"`;
    case 'INVALID_FILE_TYPE':
      return 'File type not allowed';
    case 'FILE_TOO_LARGE':
      return 'File exceeds size limit';
    case 'NO_TAGS':
      return 'At least one tag required';
    case 'INVALID_TAGS':
      return 'One or more tags not found';
    case 'STORAGE_QUOTA_EXCEEDED':
      return 'Storage limit reached';
    case 'OCR_NOT_CONFIGURED':
      return 'OCR not available';
    case 'PROCESSING_FAILED':
      return 'Document processing error';
    default:
      return typeof errorResponse.detail === 'string'
        ? errorResponse.detail
        : 'Upload failed';
  }
}
