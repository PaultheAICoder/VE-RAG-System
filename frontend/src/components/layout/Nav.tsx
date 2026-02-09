import { Link, useLocation } from 'react-router-dom';
import {
  MessageSquare,
  FileText,
  Tags,
  Users,
  Settings,
  Activity,
  Sparkles,
} from 'lucide-react';
import { useAuthStore } from '../../stores/authStore';
import type { NavItem, UserRole } from '../../types';

const navItems: NavItem[] = [
  { href: '/chat', label: 'Chat', icon: MessageSquare, roles: ['admin', 'customer_admin', 'user'] },
  { href: '/documents', label: 'Documents', icon: FileText, roles: ['admin', 'customer_admin', 'user'] },
  { href: '/tags', label: 'Tags', icon: Tags, roles: ['admin', 'customer_admin'] },
  { href: '/users', label: 'Users', icon: Users, roles: ['admin', 'customer_admin'] },
  { href: '/rag-quality', label: 'RAG Quality', icon: Sparkles, roles: ['admin'] },
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
    <nav className="sticky top-[52px] z-40 border-b border-warm-200 dark:border-warm-700 bg-white/95 dark:bg-warm-800/95 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-5">
        <div className="flex gap-0.5 overflow-x-auto hide-scrollbar">
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
                    ? 'border-primary text-primary dark:text-primary-light'
                    : 'border-transparent text-warm-500 dark:text-warm-400 hover:text-primary hover:border-warm-300 dark:hover:border-warm-600'
                  }
                `}
              >
                <Icon size={17} />
                {item.label}
              </Link>
            );
          })}
        </div>
      </div>
    </nav>
  );
}
