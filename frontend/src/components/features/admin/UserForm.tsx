import { useState, useEffect } from 'react';
import { Modal, Button, Input, Select, Checkbox } from '../../ui';
import type { UserWithTags, UserCreate, UserUpdate, UserRole } from '../../../types';

interface UserFormProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: (data: UserCreate | UserUpdate) => Promise<void>;
  user?: UserWithTags | null; // If provided, we're editing
  isLoading?: boolean;
}

const ROLE_OPTIONS = [
  { value: 'user', label: 'User' },
  { value: 'customer_admin', label: 'Customer Admin' },
  { value: 'admin', label: 'System Admin' },
];

export function UserForm({ isOpen, onClose, onSave, user, isLoading = false }: UserFormProps) {
  const [email, setEmail] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [role, setRole] = useState<UserRole>('user');
  const [tagAccessEnabled, setTagAccessEnabled] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const isEditing = Boolean(user);

  // Reset form when modal opens/closes or user changes
  useEffect(() => {
    if (isOpen) {
      if (user) {
        setEmail(user.email);
        setDisplayName(user.display_name);
        setRole(user.role);
        setTagAccessEnabled(user.tag_access_enabled ?? true);
        setPassword('');
        setConfirmPassword('');
      } else {
        setEmail('');
        setDisplayName('');
        setPassword('');
        setConfirmPassword('');
        setRole('user');
        setTagAccessEnabled(true);
      }
      setError(null);
    }
  }, [isOpen, user]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    // Validation
    if (!email.trim()) {
      setError('Email is required');
      return;
    }

    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      setError('Please enter a valid email address');
      return;
    }

    if (!displayName.trim()) {
      setError('Display name is required');
      return;
    }

    if (!isEditing) {
      if (!password) {
        setError('Password is required');
        return;
      }
      if (password.length < 8) {
        setError('Password must be at least 8 characters');
        return;
      }
      if (password !== confirmPassword) {
        setError('Passwords do not match');
        return;
      }
    }

    try {
      if (isEditing) {
        await onSave({
          email: email.trim(),
          display_name: displayName.trim(),
          role,
          tag_access_enabled: tagAccessEnabled,
        } as UserUpdate);
      } else {
        await onSave({
          email: email.trim(),
          display_name: displayName.trim(),
          password,
          role,
        } as UserCreate);
      }
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save user');
    }
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title={isEditing ? 'Edit User' : 'Create User'}
      size="md"
    >
      <form onSubmit={handleSubmit} className="space-y-4">
        {error && (
          <div className="p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg text-red-700 dark:text-red-300 text-sm">
            {error}
          </div>
        )}

        <Input
          label="Email"
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="user@example.com"
        />

        <Input
          label="Display Name"
          value={displayName}
          onChange={(e) => setDisplayName(e.target.value)}
          placeholder="John Smith"
        />

        <Select
          label="Role"
          value={role}
          onChange={(e) => setRole(e.target.value as UserRole)}
          options={ROLE_OPTIONS}
        />

        {isEditing && (
          <div>
            <Checkbox
              label="Enable tag-based access"
              checked={tagAccessEnabled}
              onChange={(e) => setTagAccessEnabled((e.target as HTMLInputElement).checked)}
            />
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1 ml-7">
              When disabled, user can access all documents regardless of tag assignments.
            </p>
          </div>
        )}

        {!isEditing && (
          <>
            <Input
              label="Password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Minimum 8 characters"
            />

            <Input
              label="Confirm Password"
              type="password"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              placeholder="Re-enter password"
            />
          </>
        )}

        {isEditing && (
          <p className="text-sm text-gray-500 dark:text-gray-400">
            To change the password, use the "Reset Password" action from the user list.
          </p>
        )}

        <div className="flex justify-end gap-3 pt-4 border-t border-gray-200 dark:border-gray-700">
          <Button variant="secondary" onClick={onClose} disabled={isLoading}>
            Cancel
          </Button>
          <Button type="submit" disabled={isLoading}>
            {isLoading ? 'Saving...' : isEditing ? 'Save Changes' : 'Create User'}
          </Button>
        </div>
      </form>
    </Modal>
  );
}
