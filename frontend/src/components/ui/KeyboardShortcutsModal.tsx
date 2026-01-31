import { Modal } from './Modal';
import { usePlatformModifier } from '../../hooks';

interface KeyboardShortcutsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

interface ShortcutItem {
  keys: string[];
  description: string;
  scope?: string;
}

export function KeyboardShortcutsModal({
  isOpen,
  onClose,
}: KeyboardShortcutsModalProps) {
  const { modSymbol } = usePlatformModifier();

  const shortcuts: ShortcutItem[] = [
    {
      keys: [modSymbol, 'K'],
      description: 'Focus search',
      scope: 'Global',
    },
    {
      keys: [modSymbol, 'D'],
      description: 'Toggle dark mode',
      scope: 'Global',
    },
    {
      keys: [modSymbol, '/'],
      description: 'Show keyboard shortcuts',
      scope: 'Global',
    },
    {
      keys: [modSymbol, 'N'],
      description: 'New chat session',
      scope: 'Chat',
    },
    {
      keys: [modSymbol, 'Enter'],
      description: 'Send message',
      scope: 'Chat Input',
    },
    {
      keys: ['Esc'],
      description: 'Close modal',
      scope: 'Any Modal',
    },
  ];

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Keyboard Shortcuts" size="md">
      <div className="space-y-4">
        <p className="text-sm text-gray-600 dark:text-gray-400">
          Use these shortcuts to navigate faster.
        </p>

        <div className="divide-y divide-gray-200 dark:divide-gray-700">
          {shortcuts.map((shortcut, index) => (
            <div
              key={index}
              className="flex items-center justify-between py-3 first:pt-0 last:pb-0"
            >
              <div className="flex-1">
                <span className="text-sm text-gray-900 dark:text-white">
                  {shortcut.description}
                </span>
                {shortcut.scope && (
                  <span className="ml-2 text-xs text-gray-500 dark:text-gray-400">
                    ({shortcut.scope})
                  </span>
                )}
              </div>
              <div className="flex items-center gap-1">
                {shortcut.keys.map((key, keyIndex) => (
                  <span key={keyIndex}>
                    <kbd className="inline-flex items-center justify-center min-w-[24px] px-2 py-1 text-xs font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm">
                      {key}
                    </kbd>
                    {keyIndex < shortcut.keys.length - 1 && (
                      <span className="mx-1 text-gray-400">+</span>
                    )}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>

        <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
          <p className="text-xs text-gray-500 dark:text-gray-400">
            Tip: Press <kbd className="px-1.5 py-0.5 text-xs bg-gray-100 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded">{modSymbol}</kbd> + <kbd className="px-1.5 py-0.5 text-xs bg-gray-100 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded">/</kbd> anytime to show this help.
          </p>
        </div>
      </div>
    </Modal>
  );
}
