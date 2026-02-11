import { useState, useRef, useCallback, useEffect } from 'react';
import { Upload, FileText } from 'lucide-react';

interface UploadDropZoneProps {
  onFilesSelected: (files: File[], skippedFiles?: File[]) => void;
  disabled?: boolean;
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
];

const ALLOWED_EXTENSIONS = ['.pdf', '.docx', '.xlsx', '.pptx', '.txt', '.md', '.html', '.csv'];

function isValidFile(file: File): boolean {
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

  while (true) {
    const batch = await new Promise<FileSystemEntry[]>((resolve, reject) => {
      reader.readEntries(resolve, reject);
    });
    if (batch.length === 0) break;
    entries.push(...batch);
  }

  return entries;
}

async function entryToFiles(entry: FileSystemEntry): Promise<File[]> {
  if (entry.isFile) {
    const fileEntry = entry as FileSystemFileEntry;
    const file = await new Promise<File>((resolve, reject) => {
      fileEntry.file(resolve, reject);
    });
    return [file];
  }

  if (entry.isDirectory) {
    const directoryEntry = entry as FileSystemDirectoryEntry;
    const childEntries = await readAllDirectoryEntries(directoryEntry);
    const nestedFiles = await Promise.all(childEntries.map((child) => entryToFiles(child)));
    return nestedFiles.flat();
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

  const fileGroups = await Promise.all(entries.map((entry) => entryToFiles(entry)));
  return fileGroups.flat();
}

export function UploadDropZone({ onFilesSelected, disabled = false }: UploadDropZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const folderInput = folderInputRef.current;
    if (!folderInput) return;
    folderInput.setAttribute('webkitdirectory', '');
    folderInput.setAttribute('directory', '');
  }, []);

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

      const files = await extractDroppedFiles(e.dataTransfer);
      const validFiles = files.filter(isValidFile);
      const skippedFiles = files.filter((file) => !isValidFile(file));

      if (validFiles.length > 0) {
        onFilesSelected(validFiles, skippedFiles);
        return;
      }

      if (skippedFiles.length > 0) {
        onFilesSelected([], skippedFiles);
      }
    },
    [disabled, onFilesSelected]
  );

  const handleClick = () => {
    if (!disabled) {
      fileInputRef.current?.click();
    }
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    const validFiles = files.filter(isValidFile);
    const skippedFiles = files.filter((file) => !isValidFile(file));

    if (validFiles.length > 0) {
      onFilesSelected(validFiles, skippedFiles);
    } else if (skippedFiles.length > 0) {
      onFilesSelected([], skippedFiles);
    }

    // Reset input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleFolderInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    const validFiles = files.filter(isValidFile);
    const skippedFiles = files.filter((file) => !isValidFile(file));

    if (validFiles.length > 0) {
      onFilesSelected(validFiles, skippedFiles);
    } else if (skippedFiles.length > 0) {
      onFilesSelected([], skippedFiles);
    }

    if (folderInputRef.current) {
      folderInputRef.current.value = '';
    }
  };

  return (
    <div
      onClick={handleClick}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className={`
        relative border-2 border-dashed rounded-xl p-8
        flex flex-col items-center justify-center
        cursor-pointer transition-all
        ${disabled
          ? 'border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50 cursor-not-allowed'
          : isDragging
            ? 'border-primary bg-primary/5 scale-[1.02]'
            : 'border-gray-300 dark:border-gray-600 hover:border-primary hover:bg-gray-50 dark:hover:bg-gray-800/50'
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
        disabled={disabled}
      />
      <input
        ref={folderInputRef}
        type="file"
        multiple
        onChange={handleFolderInputChange}
        className="hidden"
        disabled={disabled}
      />

      <div
        className={`
          w-16 h-16 rounded-full flex items-center justify-center mb-4
          ${isDragging
            ? 'bg-primary/20 text-primary'
            : 'bg-gray-100 dark:bg-gray-700 text-gray-400'
          }
        `}
      >
        {isDragging ? <FileText size={32} /> : <Upload size={32} />}
      </div>

      <p className="text-lg font-medium text-gray-700 dark:text-gray-300 mb-1">
        {isDragging ? 'Drop files here' : 'Drop files here or click to browse'}
      </p>

      <p className="text-sm text-gray-500 dark:text-gray-400">
        PDF, DOCX, XLSX, PPTX, TXT, MD, HTML, CSV
      </p>
      <button
        type="button"
        className="mt-3 text-sm text-primary hover:underline disabled:opacity-50 disabled:cursor-not-allowed"
        onClick={(e) => {
          e.stopPropagation();
          if (!disabled) {
            folderInputRef.current?.click();
          }
        }}
        disabled={disabled}
      >
        Or select a folder
      </button>
    </div>
  );
}
