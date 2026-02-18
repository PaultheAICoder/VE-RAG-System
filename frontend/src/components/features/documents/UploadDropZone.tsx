import { useState, useRef, useCallback, useEffect } from 'react';
import { Upload, FileText, Loader2 } from 'lucide-react';

interface UploadDropZoneProps {
  onFilesSelected: (files: File[], skippedFiles?: File[]) => void;
  disabled?: boolean;
  maxFiles?: number;
}

const ALLOWED_TYPES = [
  'application/pdf',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document', // .docx
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', // .xlsx
  'application/vnd.openxmlformats-officedocument.presentationml.presentation', // .pptx
  'text/plain',
  'text/markdown',
  'text/html',
  'text/csv',
  // Image formats (OCR extraction)
  'image/png',
  'image/jpeg',
  'image/tiff',
  // Email formats
  'message/rfc822', // .eml
  'application/vnd.ms-outlook', // .msg
];

const ALLOWED_EXTENSIONS = [
  '.pdf', '.docx', '.xlsx', '.pptx', '.txt', '.md', '.html', '.csv',
  // Image formats
  '.png', '.jpg', '.jpeg', '.tiff', '.tif',
  // Email formats
  '.eml', '.msg',
];

// Hidden/system files to always skip
const IGNORED_NAMES = new Set(['.DS_Store', 'Thumbs.db', 'desktop.ini', '.gitkeep']);
const IGNORED_PREFIXES = ['~$', '._'];

function isHiddenOrSystemFile(name: string): boolean {
  if (IGNORED_NAMES.has(name)) return true;
  if (IGNORED_PREFIXES.some((p) => name.startsWith(p))) return true;
  // Files inside __MACOSX directory
  if (name.includes('__MACOSX')) return true;
  return false;
}

function isValidFile(file: File): boolean {
  if (isHiddenOrSystemFile(file.name)) return false;
  // Check MIME type
  if (ALLOWED_TYPES.includes(file.type)) {
    return true;
  }
  // Also check extension for cases where MIME type is not set correctly
  const ext = '.' + file.name.split('.').pop()?.toLowerCase();
  return ALLOWED_EXTENSIONS.includes(ext);
}

async function readAllDirectoryEntries(
  directory: FileSystemDirectoryEntry
): Promise<FileSystemEntry[]> {
  const reader = directory.createReader();
  const entries: FileSystemEntry[] = [];

  try {
    while (true) {
      const batch = await new Promise<FileSystemEntry[]>((resolve, reject) => {
        reader.readEntries(resolve, reject);
      });
      if (batch.length === 0) break;
      entries.push(...batch);
    }
  } catch (err) {
    console.warn(`Failed to read directory "${directory.name}":`, err);
  }

  return entries;
}

async function entryToFiles(entry: FileSystemEntry): Promise<File[]> {
  if (entry.isFile) {
    const fileEntry = entry as FileSystemFileEntry;
    try {
      const file = await new Promise<File>((resolve, reject) => {
        fileEntry.file(resolve, reject);
      });
      return [file];
    } catch (err) {
      console.warn(`Failed to read file "${entry.name}":`, err);
      return [];
    }
  }

  if (entry.isDirectory) {
    const directoryEntry = entry as FileSystemDirectoryEntry;
    const childEntries = await readAllDirectoryEntries(directoryEntry);
    const results = await Promise.allSettled(childEntries.map((child) => entryToFiles(child)));
    return results
      .filter((r): r is PromiseFulfilledResult<File[]> => r.status === 'fulfilled')
      .flatMap((r) => r.value);
  }

  return [];
}

async function extractDroppedFiles(dataTransfer: DataTransfer): Promise<File[]> {
  const items = dataTransfer.items ? Array.from(dataTransfer.items) : [];
  const entries = items
    .map((item) => item.webkitGetAsEntry())
    .filter((entry): entry is FileSystemEntry => entry !== null);

  if (entries.length === 0) {
    return Array.from(dataTransfer.files);
  }

  const results = await Promise.allSettled(entries.map((entry) => entryToFiles(entry)));
  return results
    .filter((r): r is PromiseFulfilledResult<File[]> => r.status === 'fulfilled')
    .flatMap((r) => r.value);
}

const DEFAULT_MAX_FILES = 500;

export function UploadDropZone({
  onFilesSelected,
  disabled = false,
  maxFiles = DEFAULT_MAX_FILES,
}: UploadDropZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [isEnumerating, setIsEnumerating] = useState(false);
  const [enumerationError, setEnumerationError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);
  const lastPickerOpenAtRef = useRef(0);

  const configureFolderInput = useCallback(() => {
    const folderInput = folderInputRef.current;
    if (!folderInput) return;
    folderInput.setAttribute('webkitdirectory', '');
    folderInput.setAttribute('directory', '');
    // Enforce single-folder selection semantics.
    folderInput.removeAttribute('multiple');
  }, []);

  useEffect(() => {
    configureFolderInput();
  }, [configureFolderInput]);

  const processFiles = useCallback(
    (files: File[]) => {
      setEnumerationError(null);

      const validFiles = files.filter(isValidFile);
      const skippedFiles = files.filter((f) => !isHiddenOrSystemFile(f.name) && !isValidFile(f));

      if (validFiles.length > maxFiles) {
        setEnumerationError(
          `Too many files (${validFiles.length}). Maximum is ${maxFiles}. Select a smaller folder.`
        );
        return;
      }

      if (validFiles.length > 0) {
        onFilesSelected(validFiles, skippedFiles);
      } else if (skippedFiles.length > 0) {
        onFilesSelected([], skippedFiles);
      }
    },
    [maxFiles, onFilesSelected]
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (!disabled) {
      setIsDragging(true);
    }
  }, [disabled]);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    async (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(false);

      if (disabled) return;

      setIsEnumerating(true);
      setEnumerationError(null);
      try {
        const files = await extractDroppedFiles(e.dataTransfer);
        processFiles(files);
      } catch (err) {
        console.error('Failed to read dropped files:', err);
        setEnumerationError('Failed to read some files. Try using "Select folder" instead.');
      } finally {
        setIsEnumerating(false);
      }
    },
    [disabled, processFiles]
  );

  const openPicker = useCallback(
    (picker: 'file' | 'folder') => {
      if (disabled || isEnumerating) return;

      const now = Date.now();
      if (now - lastPickerOpenAtRef.current < 500) {
        return;
      }
      lastPickerOpenAtRef.current = now;

      setEnumerationError(null);

      if (picker === 'folder') {
        configureFolderInput();
        folderInputRef.current?.click();
        return;
      }

      fileInputRef.current?.click();
    },
    [configureFolderInput, disabled, isEnumerating]
  );

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    processFiles(files);

    // Reset input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleFolderInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    setIsEnumerating(true);
    try {
      processFiles(files);
    } finally {
      setIsEnumerating(false);
    }

    if (folderInputRef.current) {
      folderInputRef.current.value = '';
    }
  };

  const isDisabled = disabled || isEnumerating;

  return (
    <div
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className={`
        relative border-2 border-dashed rounded-xl p-8
        flex flex-col items-center justify-center
        transition-all
        ${isDisabled
          ? 'border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50'
          : isDragging
            ? 'border-primary bg-primary/5 scale-[1.02]'
            : 'border-gray-300 dark:border-gray-600'
        }
      `}
    >
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept={ALLOWED_EXTENSIONS.join(',')}
        onChange={handleFileInputChange}
        className="hidden"
        disabled={isDisabled}
      />
      <input
        ref={folderInputRef}
        type="file"
        onChange={handleFolderInputChange}
        className="hidden"
        disabled={isDisabled}
      />

      <div
        className={`
          w-16 h-16 rounded-full flex items-center justify-center mb-4
          ${isEnumerating
            ? 'bg-primary/20 text-primary'
            : isDragging
              ? 'bg-primary/20 text-primary'
              : 'bg-gray-100 dark:bg-gray-700 text-gray-400'
          }
        `}
      >
        {isEnumerating ? (
          <Loader2 size={32} className="animate-spin" />
        ) : isDragging ? (
          <FileText size={32} />
        ) : (
          <Upload size={32} />
        )}
      </div>

      <p className="text-lg font-medium text-gray-700 dark:text-gray-300 mb-1">
        {isEnumerating
          ? 'Reading folder contents...'
          : isDragging
            ? 'Drop files here'
            : 'Drag and drop files or folders here'}
      </p>

      {enumerationError ? (
        <p className="text-sm text-red-500 dark:text-red-400 mt-1 text-center max-w-md">
          {enumerationError}
        </p>
      ) : (
        <p className="text-sm text-gray-500 dark:text-gray-400">
          PDF, DOCX, XLSX, PPTX, TXT, MD, HTML, CSV, PNG, JPG, TIFF, EML, MSG
        </p>
      )}
      <div className="mt-4 flex flex-wrap items-center justify-center gap-3">
        <button
          type="button"
          className="px-3 py-1.5 text-sm rounded-md border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
          onClick={() => openPicker('file')}
          disabled={isDisabled}
        >
          Select files
        </button>
        <button
          type="button"
          className="px-3 py-1.5 text-sm rounded-md border border-primary text-primary hover:bg-primary/5 disabled:opacity-50 disabled:cursor-not-allowed"
          onClick={() => openPicker('folder')}
          disabled={isDisabled}
        >
          Select folder
        </button>
      </div>
    </div>
  );
}
