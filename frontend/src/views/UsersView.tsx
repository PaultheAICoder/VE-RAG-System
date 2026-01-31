import { useState, useEffect, useCallback } from 'react';
import { Plus, Pencil, Key, Tags, UserX, Search } from 'lucide-react';
import { Button, Alert, Card, Input, Select, Badge } from '../components/ui';
import {
  UserForm,
  TagAssignmentModal,
  PasswordResetModal,
  ConfirmModal,
} from '../components/features/admin';
import {
  listUsers,
  createUser,
  updateUser,
  deactivateUser,
  assignUserTags,
  resetUserPassword,
} from '../api/users';
import { listTags } from '../api/tags';
import type { UserWithTags, UserCreate, UserUpdate, Tag, UserRole } from '../types';

const ROLE_OPTIONS = [
  { value: '', label: 'All Roles' },
  { value: 'user', label: 'User' },
  { value: 'customer_admin', label: 'Customer Admin' },
  { value: 'admin', label: 'System Admin' },
];

const ROLE_LABELS: Record<UserRole, string> = {
  user: 'User',
  customer_admin: 'Customer Admin',
  admin: 'System Admin',
};

const ROLE_COLORS: Record<UserRole, 'default' | 'primary' | 'success' | 'warning' | 'danger'> = {
  user: 'default',
  customer_admin: 'primary',
  admin: 'success',
};

export function UsersView() {
  // Data state
  const [users, setUsers] = useState<UserWithTags[]>([]);
  const [tags, setTags] = useState<Tag[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Filter state
  const [search, setSearch] = useState('');
  const [roleFilter, setRoleFilter] = useState('');

  // Modal state
  const [showUserForm, setShowUserForm] = useState(false);
  const [editingUser, setEditingUser] = useState<UserWithTags | null>(null);
  const [assigningTagsUser, setAssigningTagsUser] = useState<UserWithTags | null>(null);
  const [deactivatingUser, setDeactivatingUser] = useState<UserWithTags | null>(null);
  const [resetPasswordResult, setResetPasswordResult] = useState<{
    userName: string;
    password: string;
  } | null>(null);
  const [actionLoading, setActionLoading] = useState(false);

  // Fetch users
  const fetchUsers = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await listUsers();
      setUsers(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load users');
    } finally {
      setLoading(false);
    }
  }, []);

  // Fetch tags
  const fetchTags = useCallback(async () => {
    try {
      const data = await listTags();
      setTags(data);
    } catch (err) {
      console.error('Failed to fetch tags:', err);
    }
  }, []);

  useEffect(() => {
    fetchUsers();
    fetchTags();
  }, [fetchUsers, fetchTags]);

  // Filter users
  const filteredUsers = users.filter((user) => {
    const matchesSearch =
      !search ||
      user.email.toLowerCase().includes(search.toLowerCase()) ||
      user.display_name.toLowerCase().includes(search.toLowerCase());
    const matchesRole = !roleFilter || user.role === roleFilter;
    return matchesSearch && matchesRole;
  });

  // Handle create user
  const handleCreate = () => {
    setEditingUser(null);
    setShowUserForm(true);
  };

  // Handle edit user
  const handleEdit = (user: UserWithTags) => {
    setEditingUser(user);
    setShowUserForm(true);
  };

  // Handle save user
  const handleSaveUser = async (data: UserCreate | UserUpdate) => {
    setActionLoading(true);
    try {
      if (editingUser) {
        await updateUser(editingUser.id, data as UserUpdate);
      } else {
        await createUser(data as UserCreate);
      }
      await fetchUsers();
      setShowUserForm(false);
    } catch (err) {
      throw err;
    } finally {
      setActionLoading(false);
    }
  };

  // Handle assign tags
  const handleAssignTags = (user: UserWithTags) => {
    setAssigningTagsUser(user);
  };

  const handleSaveTags = async (tagIds: string[]) => {
    if (!assigningTagsUser) return;
    setActionLoading(true);
    try {
      await assignUserTags(assigningTagsUser.id, tagIds);
      await fetchUsers();
      setAssigningTagsUser(null);
    } catch (err) {
      throw err;
    } finally {
      setActionLoading(false);
    }
  };

  // Handle reset password
  const handleResetPassword = async (user: UserWithTags) => {
    setActionLoading(true);
    try {
      const result = await resetUserPassword(user.id);
      setResetPasswordResult({
        userName: user.display_name,
        password: result.temporary_password,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to reset password');
    } finally {
      setActionLoading(false);
    }
  };

  // Handle deactivate user
  const handleDeactivate = (user: UserWithTags) => {
    setDeactivatingUser(user);
  };

  const handleConfirmDeactivate = async () => {
    if (!deactivatingUser) return;
    setActionLoading(true);
    try {
      await deactivateUser(deactivatingUser.id);
      await fetchUsers();
      setDeactivatingUser(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to deactivate user');
    } finally {
      setActionLoading(false);
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-heading font-bold text-gray-900 dark:text-white">Users</h1>
          <p className="text-gray-500 dark:text-gray-400 mt-1">
            Manage user accounts and permissions
          </p>
        </div>
        <Button icon={Plus} onClick={handleCreate}>
          Add User
        </Button>
      </div>

      {/* Error Alert */}
      {error && (
        <Alert variant="danger" onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Filters */}
      <div className="flex gap-4">
        <Input
          placeholder="Search users..."
          icon={Search}
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="max-w-sm"
        />
        <Select
          options={ROLE_OPTIONS}
          value={roleFilter}
          onChange={(e) => setRoleFilter(e.target.value)}
          className="w-48"
        />
      </div>

      {/* Users Table */}
      <Card>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-200 dark:border-gray-700">
                <th className="text-left py-3 px-4 text-sm font-semibold text-gray-600 dark:text-gray-300">
                  Name
                </th>
                <th className="text-left py-3 px-4 text-sm font-semibold text-gray-600 dark:text-gray-300">
                  Email
                </th>
                <th className="text-left py-3 px-4 text-sm font-semibold text-gray-600 dark:text-gray-300">
                  Role
                </th>
                <th className="text-left py-3 px-4 text-sm font-semibold text-gray-600 dark:text-gray-300">
                  Tags
                </th>
                <th className="text-left py-3 px-4 text-sm font-semibold text-gray-600 dark:text-gray-300">
                  Status
                </th>
                <th className="text-right py-3 px-4 text-sm font-semibold text-gray-600 dark:text-gray-300">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan={6} className="py-8 text-center text-gray-500">
                    Loading users...
                  </td>
                </tr>
              ) : filteredUsers.length === 0 ? (
                <tr>
                  <td colSpan={6} className="py-8 text-center text-gray-500">
                    {users.length === 0
                      ? 'No users found. Create your first user to get started.'
                      : 'No users match your filters.'}
                  </td>
                </tr>
              ) : (
                filteredUsers.map((user) => (
                  <tr
                    key={user.id}
                    className={`border-b border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-800/50 ${
                      !user.is_active ? 'opacity-50' : ''
                    }`}
                  >
                    <td className="py-3 px-4">
                      <span className="font-medium text-gray-900 dark:text-white">
                        {user.display_name}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-gray-600 dark:text-gray-400">{user.email}</td>
                    <td className="py-3 px-4">
                      <Badge variant={ROLE_COLORS[user.role]}>{ROLE_LABELS[user.role]}</Badge>
                    </td>
                    <td className="py-3 px-4">
                      <div className="flex flex-wrap gap-1">
                        {user.tags.length === 0 ? (
                          <span className="text-gray-400 text-sm">No tags</span>
                        ) : user.role === 'admin' || user.role === 'customer_admin' ? (
                          <Badge variant="primary">All</Badge>
                        ) : (
                          user.tags.slice(0, 3).map((tag) => (
                            <Badge key={tag.id}>{tag.display_name}</Badge>
                          ))
                        )}
                        {user.tags.length > 3 &&
                          user.role !== 'admin' &&
                          user.role !== 'customer_admin' && (
                            <Badge>+{user.tags.length - 3}</Badge>
                          )}
                      </div>
                    </td>
                    <td className="py-3 px-4">
                      {user.is_active ? (
                        <Badge variant="success">Active</Badge>
                      ) : (
                        <Badge variant="danger">Inactive</Badge>
                      )}
                    </td>
                    <td className="py-3 px-4">
                      <div className="flex items-center justify-end gap-1">
                        <button
                          onClick={() => handleEdit(user)}
                          className="p-1.5 text-gray-400 hover:text-primary hover:bg-primary/10 rounded transition-colors"
                          title="Edit user"
                        >
                          <Pencil size={16} />
                        </button>
                        <button
                          onClick={() => handleAssignTags(user)}
                          className="p-1.5 text-gray-400 hover:text-primary hover:bg-primary/10 rounded transition-colors"
                          title="Assign tags"
                        >
                          <Tags size={16} />
                        </button>
                        <button
                          onClick={() => handleResetPassword(user)}
                          className="p-1.5 text-gray-400 hover:text-amber-500 hover:bg-amber-50 dark:hover:bg-amber-900/20 rounded transition-colors"
                          title="Reset password"
                          disabled={actionLoading}
                        >
                          <Key size={16} />
                        </button>
                        {user.is_active && (
                          <button
                            onClick={() => handleDeactivate(user)}
                            className="p-1.5 text-gray-400 hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20 rounded transition-colors"
                            title="Deactivate user"
                          >
                            <UserX size={16} />
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </Card>

      {/* User Form Modal */}
      <UserForm
        isOpen={showUserForm}
        onClose={() => {
          setShowUserForm(false);
          setEditingUser(null);
        }}
        onSave={handleSaveUser}
        user={editingUser}
        isLoading={actionLoading}
      />

      {/* Tag Assignment Modal */}
      <TagAssignmentModal
        isOpen={Boolean(assigningTagsUser)}
        onClose={() => setAssigningTagsUser(null)}
        onSave={handleSaveTags}
        user={assigningTagsUser}
        availableTags={tags}
        isLoading={actionLoading}
      />

      {/* Password Reset Modal */}
      {resetPasswordResult && (
        <PasswordResetModal
          isOpen={Boolean(resetPasswordResult)}
          onClose={() => setResetPasswordResult(null)}
          userName={resetPasswordResult.userName}
          temporaryPassword={resetPasswordResult.password}
        />
      )}

      {/* Deactivate Confirmation Modal */}
      <ConfirmModal
        isOpen={Boolean(deactivatingUser)}
        onClose={() => setDeactivatingUser(null)}
        onConfirm={handleConfirmDeactivate}
        title="Deactivate User"
        message={`Are you sure you want to deactivate ${deactivatingUser?.display_name}? They will no longer be able to log in.`}
        confirmLabel="Deactivate"
        isLoading={actionLoading}
        variant="danger"
      />
    </div>
  );
}
