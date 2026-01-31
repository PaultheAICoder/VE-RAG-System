import { Link, useLocation } from 'react-router-dom';
import {
  MessageSquare,
  FileText,
  Tags,
  Users,
  Settings,
  Activity,
} from 'lucide-react';
import { useAuthStore } from '../../stores/authStore';
import type { NavItem, UserRole } from '../../types';

const navItems: NavItem[] = [
  { href: '/chat', label: 'Chat', icon: MessageSquare, roles: ['admin', 'customer_admin', 'user'] },
  { href: '/documents', label: 'Documents', icon: FileText, roles: ['admin', 'customer_admin', 'user'] },
  { href: '/tags', label: 'Tags', icon: Tags, roles: ['admin', 'customer_admin'] },
  { href: '/users', label: 'Users', icon: Users, roles: ['admin', 'customer_admin'] },
  { href: '/settings', label: 'Settings', icon: Settings, roles: ['admin'] },
  { href: '/health', label: 'Health', icon: Activity, roles: ['admin'] },
];

export function Nav() {
  const location = useLocation();
  const { user } = useAuthStore();
  const userRole = user?.role as UserRole;

  const visibleItems = navItems.filter((item) =>
    item.roles.includes(userRole)
  );

  return (
    <nav className="border-b border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900">
      <div className="max-w-7xl mx-auto px-4">
        <div className="flex gap-1 overflow-x-auto">
          {visibleItems.map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.href;

            return (
              <Link
                key={item.href}
                to={item.href}
                className={`
                  flex items-center gap-2 px-4 py-3 text-sm font-medium
                  border-b-2 -mb-px whitespace-nowrap transition-colors
                  ${isActive
                    ? 'border-primary text-primary'
                    : 'border-transparent text-gray-600 dark:text-gray-400 hover:text-primary hover:border-gray-300'
                  }
                `}
              >
                <Icon size={18} />
                {item.label}
              </Link>
            );
          })}
        </div>
      </div>
    </nav>
  );
}
